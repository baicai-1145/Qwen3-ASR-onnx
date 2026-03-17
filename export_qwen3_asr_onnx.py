#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
import onnx
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoProcessor

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
DEFAULT_EOS_TOKEN_IDS = [151645, 151643]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a full single-audio ONNX inference pipeline for Qwen3-ASR. "
            "Export phase uses PyTorch, but runtime can stay on ONNX Runtime + tokenizer/feature extractor."
        )
    )
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for ONNX files and metadata.")
    parser.add_argument("--device", default="cuda:0", help="Export device.")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16", help="Export dtype.")
    parser.add_argument("--sample-seconds", type=float, default=1.0, help="Dummy audio length used for tracing.")
    parser.add_argument("--static-cache-len", type=int, default=1664, help="Fixed KV cache length for exported decoder graph.")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ONNX files.")
    parser.add_argument("--worklog", default="WORKLOG.md", help="Optional markdown log.")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32}[name]


def feature_output_lengths(lengths: torch.Tensor) -> torch.Tensor:
    leave = lengths % 100
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (lengths // 100) * 13


def build_prompt(processor: AutoProcessor) -> str:
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def flatten_legacy_cache(cache: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for key_states, value_states in cache:
        flat.append(key_states)
        flat.append(value_states)
    return flat


class FixedSizeCacheLayer:
    def __init__(self, max_cache_len: int):
        self.max_cache_len = max_cache_len
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def allocate_like(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.keys = torch.zeros(
            (key_states.shape[0], key_states.shape[1], self.max_cache_len, key_states.shape[3]),
            dtype=key_states.dtype,
            device=key_states.device,
        )
        self.values = torch.zeros(
            (value_states.shape[0], value_states.shape[1], self.max_cache_len, value_states.shape[3]),
            dtype=value_states.dtype,
            device=value_states.device,
        )

    def load(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.keys = key_states
        self.values = value_states

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: dict[str, torch.Tensor] | None):
        if self.keys is None or self.values is None:
            self.allocate_like(key_states, value_states)
        cache_position = cache_kwargs["cache_position"] if cache_kwargs is not None else None
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[-2], device=key_states.device)
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        visible_len = cache_position[-1] + 1
        return self.keys[:, :, :visible_len, :], self.values[:, :, :visible_len, :]


class FixedSizeCache:
    def __init__(self, num_layers: int, max_cache_len: int):
        self.layers = [FixedSizeCacheLayer(max_cache_len=max_cache_len) for _ in range(num_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, torch.Tensor] | None = None,
    ):
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)


def flatten_static_cache(cache: FixedSizeCache) -> List[torch.Tensor]:
    flat: List[torch.Tensor] = []
    for layer in cache.layers:
        flat.append(layer.keys)
        flat.append(layer.values)
    return flat


def export_graph(
    module: nn.Module,
    args_tuple: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    existing_files = {path.name for path in output_path.parent.iterdir() if path.is_file()}
    torch.onnx.export(
        module,
        args=args_tuple,
        f=str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        dynamo=False,
        external_data=True,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    data_path = output_path.with_suffix(output_path.suffix + ".data")
    model = onnx.load(str(output_path), load_external_data=True)
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_path.name,
        size_threshold=0,
        convert_attribute=False,
    )
    current_files = [path for path in output_path.parent.iterdir() if path.is_file()]
    keep_names = existing_files | {output_path.name, data_path.name}
    for path in current_files:
        if path.name not in keep_names:
            path.unlink()


class ConvEmbedCore(nn.Module):
    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower

    def forward(self, padded_feature: torch.Tensor):
        x = torch.nn.functional.gelu(self.audio_tower.conv2d1(padded_feature))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d2(x))
        x = torch.nn.functional.gelu(self.audio_tower.conv2d3(x))
        batch_size, channels, freq_bins, time_steps = x.shape
        x = self.audio_tower.conv_out(
            x.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, channels * freq_bins)
        )
        pos = self.audio_tower.positional_embedding.positional_embedding[: x.shape[1], :].unsqueeze(0).to(x.dtype)
        return x + pos


class AudioStackCore(nn.Module):
    def __init__(self, audio_tower: nn.Module):
        super().__init__()
        self.audio_tower = audio_tower
        sample_attn = audio_tower.layers[0].self_attn
        self.num_heads = sample_attn.num_heads
        self.head_dim = sample_attn.head_dim
        self.scaling = sample_attn.scaling

    def self_attention(self, attn_mod: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        query = attn_mod.q_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        key = attn_mod.k_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        value = attn_mod.v_proj(hidden_states).view(seq_len, self.num_heads, self.head_dim).transpose(0, 1).unsqueeze(0)
        attn_weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).reshape(seq_len, self.num_heads * self.head_dim).contiguous()
        return attn_mod.out_proj(attn_output)

    def forward(self, hidden_states: torch.Tensor):
        x = hidden_states
        for layer in self.audio_tower.layers:
            residual = x
            x = layer.self_attn_layer_norm(x)
            x = self.self_attention(layer.self_attn, x)
            x = residual + x

            residual = x
            x = layer.final_layer_norm(x)
            x = layer.fc1(x)
            x = layer.activation_fn(x)
            x = layer.fc2(x)
            x = residual + x

            if x.dtype == torch.float16:
                clamp_value = torch.finfo(x.dtype).max - 1000
                x = torch.clamp(x, min=-clamp_value, max=clamp_value)

        x = self.audio_tower.ln_post(x)
        x = self.audio_tower.proj1(x)
        x = self.audio_tower.act(x)
        x = self.audio_tower.proj2(x)
        return x


class DecoderCore(nn.Module):
    def __init__(self, thinker: nn.Module, audio_token_id: int, num_layers: int, max_cache_len: int):
        super().__init__()
        self.thinker = thinker
        self.text_model = thinker.model
        self.audio_token_id = audio_token_id
        self.num_layers = num_layers
        self.max_cache_len = max_cache_len

    def build_static_cache(self, past_key_values: Sequence[torch.Tensor]) -> FixedSizeCache:
        cache = FixedSizeCache(num_layers=self.num_layers, max_cache_len=self.max_cache_len)
        for layer_idx, layer in enumerate(cache.layers):
            key_states = past_key_values[layer_idx * 2]
            value_states = past_key_values[layer_idx * 2 + 1]
            layer.load(key_states, value_states)
        return cache

    def build_inputs_embeds(self, input_ids: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.thinker.get_input_embeddings()(input_ids)
        audio_mask = (input_ids == self.audio_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(audio_mask, audio_features.to(inputs_embeds.dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        cache_position: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        cache = self.build_static_cache(past_key_values)
        inputs_embeds = self.build_inputs_embeds(input_ids, audio_features)
        position_ids = cache_position.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        text_position_ids = position_ids[0]
        visible_len = cache_position[-1] + 1

        key_positions = torch.arange(visible_len, device=input_ids.device)
        query_positions = cache_position.view(-1, 1)
        blocked = key_positions.view(1, visible_len) > query_positions
        causal_mask = torch.zeros(
            (input_ids.shape[0], 1, input_ids.shape[1], visible_len),
            dtype=inputs_embeds.dtype,
            device=input_ids.device,
        )
        min_value = torch.finfo(inputs_embeds.dtype).min
        causal_mask = causal_mask.masked_fill(blocked.unsqueeze(0).unsqueeze(0), min_value)

        hidden_states = inputs_embeds
        position_embeddings = self.text_model.rotary_emb(hidden_states, position_ids)
        for layer in self.text_model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=text_position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.text_model.norm(hidden_states)
        last_hidden_state = hidden_states[:, -1:, :]
        logits = self.thinker.lm_head(last_hidden_state)
        return (logits[:, 0, :], *flatten_static_cache(cache))


def ensure_overwrite_allowed(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing files: {joined}. Use --overwrite to replace them.")


def append_worklog(worklog_path: Path, model_name: str, output_dir: Path, dtype_name: str) -> None:
    lines = [
        "",
        f"### 节点 6：ONNX 导出完成（{model_name}）",
        f"- 完成时间：`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "- 导出范围：单音频全链路纯 ONNX 运行时所需的 3 张图：`conv_embed`、`audio_stack`、`decoder`。",
        "- 运行时依赖目标：`onnxruntime + transformers(tokenizer/feature_extractor) + numpy + soundfile/librosa`，不依赖 PyTorch。",
        "- 说明：为绕过上游音频编码链路的导出器问题，音频 encoder stack 使用同权重的 ONNX 友好实现。",
        f"- 导出 dtype：`{dtype_name}`",
        f"- 导出目录：`{output_dir}`",
        "- decoder 图说明：prefill/decode 复用同一张 decoder 图与同一套权重，保留固定 shape 静态 KV cache。",
    ]
    with worklog_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        output_dir / "conv_embed.onnx",
        output_dir / "audio_stack.onnx",
        output_dir / "decoder.onnx",
        output_dir / "metadata.json",
    ]
    ensure_overwrite_allowed(paths, overwrite=args.overwrite)

    export_dtype = resolve_dtype(args.dtype)
    model = AutoModel.from_pretrained(
        str(model_path),
        dtype=export_dtype,
        device_map=args.device,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(str(model_path), fix_mistral_regex=True)
    thinker = model.thinker.eval()
    audio_tower = thinker.audio_tower.eval()
    text_config = thinker.config.text_config
    audio_config = thinker.config.audio_config

    prompt = build_prompt(processor)
    dummy_audio = np.zeros(int(args.sample_seconds * 16000), dtype=np.float32)
    inputs = processor(text=[prompt], audio=[dummy_audio], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(args.device)
    input_features = inputs["input_features"].to(device=args.device, dtype=export_dtype)[0]
    feature_attention_mask = inputs["feature_attention_mask"].to(args.device)[0]
    feature_len = int(feature_attention_mask.sum().item())

    window = audio_config.n_window * 2
    chunk_num = (feature_len + window - 1) // window
    padded_total = chunk_num * window
    feature_slice = input_features[:, :feature_len]
    if padded_total > feature_len:
        feature_slice = torch.nn.functional.pad(feature_slice, (0, padded_total - feature_len))
    padded_feature = (
        feature_slice.transpose(0, 1)
        .reshape(chunk_num, window, feature_slice.shape[0])
        .transpose(1, 2)
        .unsqueeze(1)
        .contiguous()
    )
    chunk_lengths = torch.full((chunk_num,), window, device=input_features.device, dtype=torch.long)
    chunk_lengths[-1] = feature_len - window * (chunk_num - 1)
    chunk_feature_lengths_after = feature_output_lengths(chunk_lengths)

    conv_embed_core = ConvEmbedCore(audio_tower).eval()
    with torch.no_grad():
        padded_embed = conv_embed_core(padded_feature)
        mask_after = (
            torch.arange(padded_embed.shape[1], device=padded_embed.device).unsqueeze(0)
            < chunk_feature_lengths_after.unsqueeze(1)
        )
        hidden_states = padded_embed[mask_after]
        audio_stack_core = AudioStackCore(audio_tower).eval()
        audio_features = audio_stack_core(hidden_states)

    export_graph(
        conv_embed_core,
        args_tuple=(padded_feature,),
        output_path=output_dir / "conv_embed.onnx",
        input_names=["padded_feature"],
        output_names=["padded_embed"],
        dynamic_axes={
            "padded_feature": {0: "num_chunks", 3: "chunk_frames"},
            "padded_embed": {0: "num_chunks", 1: "chunk_frames_after_cnn"},
        },
        opset=args.opset,
    )

    export_graph(
        audio_stack_core,
        args_tuple=(hidden_states,),
        output_path=output_dir / "audio_stack.onnx",
        input_names=["hidden_states"],
        output_names=["audio_features"],
        dynamic_axes={
            "hidden_states": {0: "audio_seq"},
            "audio_features": {0: "audio_seq"},
        },
        opset=args.opset,
    )

    sample_attn = thinker.model.layers[0].self_attn
    num_layers = len(thinker.model.layers)
    num_key_value_heads = int(sample_attn.config.num_key_value_heads)
    head_dim = int(sample_attn.head_dim)
    present_names = [f"present_key_{i:02d}" if p == 0 else f"present_value_{i:02d}" for i in range(num_layers) for p in range(2)]
    past_names = [f"past_key_{i:02d}" if p == 0 else f"past_value_{i:02d}" for i in range(num_layers) for p in range(2)]
    empty_cache = tuple(
        torch.zeros(
            (1, num_key_value_heads, args.static_cache_len, head_dim),
            device=input_ids.device,
            dtype=export_dtype,
        )
        for _ in range(num_layers * 2)
    )
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
    decoder_core = DecoderCore(
        thinker,
        audio_token_id=thinker.config.audio_token_id,
        num_layers=num_layers,
        max_cache_len=args.static_cache_len,
    ).eval()
    export_graph(
        decoder_core,
        args_tuple=(input_ids, audio_features, cache_position, *empty_cache),
        output_path=output_dir / "decoder.onnx",
        input_names=["input_ids", "audio_features", "cache_position", *past_names],
        output_names=["logits", *present_names],
        dynamic_axes={
            "input_ids": {1: "query_seq"},
            "audio_features": {0: "audio_seq"},
            "cache_position": {0: "query_seq"},
            "logits": {0: "batch"},
        },
        opset=args.opset,
    )

    metadata = {
        "model_path": str(model_path),
        "model_name": model_path.name,
        "dtype": args.dtype,
        "device": args.device,
        "opset": args.opset,
        "eos_token_ids": DEFAULT_EOS_TOKEN_IDS,
        "sample_seconds": args.sample_seconds,
        "runtime_mode": "single_audio_full_onnx_no_torch",
        "audio_token_id": int(thinker.config.audio_token_id),
        "audio_token": processor.tokenizer.audio_token,
        "feature_size": int(audio_config.num_mel_bins),
        "n_window": int(audio_config.n_window),
        "n_window_infer": int(audio_config.n_window_infer),
        "audio_d_model": int(audio_config.d_model),
        "audio_output_dim": int(audio_config.output_dim),
        "text_hidden_size": int(text_config.hidden_size),
        "num_layers": num_layers,
        "static_cache_len": int(args.static_cache_len),
        "conv_embed_model": "conv_embed.onnx",
        "audio_stack_model": "audio_stack.onnx",
        "decoder_model": "decoder.onnx",
        "prefill_model": "decoder.onnx",
        "decode_model": "decoder.onnx",
        "shared_decoder_session": True,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if args.worklog:
        append_worklog(Path(args.worklog).resolve(), model_name=model_path.name, output_dir=output_dir, dtype_name=args.dtype)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
