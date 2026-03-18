#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import itertools
import json
import math
import queue
import threading
import time
import urllib.request
from concurrent.futures import Future
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer, WhisperFeatureExtractor

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import librosa
except Exception:
    librosa = None

try:
    from scipy.signal import resample_poly
except Exception:
    resample_poly = None


def is_url(text: str) -> bool:
    parsed = urlparse(text)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def is_probably_base64(text: str) -> bool:
    return text.startswith("data:audio") or ("/" not in text and "\\" not in text and len(text) > 256)


def decode_base64_bytes(text: str) -> bytes:
    if "," in text and text.strip().startswith("data:"):
        text = text.split(",", 1)[1]
    return base64.b64decode(text)


def load_audio_any(value: str) -> Tuple[np.ndarray, int]:
    if is_url(value):
        request = urllib.request.Request(value, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()
        with io.BytesIO(data) as buffer:
            audio, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
    elif is_probably_base64(value):
        data = decode_base64_bytes(value)
        with io.BytesIO(data) as buffer:
            audio, sample_rate = sf.read(buffer, dtype="float32", always_2d=False)
    else:
        audio, sample_rate = librosa.load(value, sr=None, mono=False)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def normalize_audio_waveform(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = to_mono(np.asarray(audio, dtype=np.float32))
    if sample_rate != 16000:
        if librosa is not None:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=16000).astype(np.float32)
        elif resample_poly is not None:
            scale = gcd(sample_rate, 16000)
            mono = resample_poly(mono, 16000 // scale, sample_rate // scale).astype(np.float32)
        else:
            raise RuntimeError("Resampling requires either librosa or scipy.")
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak
    return np.clip(mono, -1.0, 1.0).astype(np.float32)


def normalize_audio_input(value: str) -> np.ndarray:
    audio, sample_rate = load_audio_any(value)
    return normalize_audio_waveform(audio, sample_rate)


def feature_output_lengths(lengths: np.ndarray) -> np.ndarray:
    leave = lengths % 100
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (lengths // 100) * 13


def parse_asr_output(raw: str, user_language: str | None = None) -> Tuple[str, str]:
    if raw is None:
        return "", ""
    text = str(raw).strip()
    if not text:
        return "", ""
    if user_language:
        return user_language, text
    if "<asr_text>" not in text:
        return "", text
    meta, text_part = text.split("<asr_text>", 1)
    meta_lower = meta.lower()
    if "language none" in meta_lower:
        return "", text_part.strip()
    language = ""
    for line in meta.splitlines():
        line = line.strip()
        if line.lower().startswith("language "):
            language = line[len("language ") :].strip()
            break
    return language, text_part.strip()


def load_session(path: Path, providers: List[str]) -> ort.InferenceSession:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), providers=providers, sess_options=session_options)


def run_session(session: ort.InferenceSession, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    output_items = session.get_outputs()
    values = session.run([item.name for item in output_items], inputs)
    return dict(zip([item.name for item in output_items], values))


def ortvector_tail(vector, start: int) -> List[ort.OrtValue]:
    return [vector[index] for index in range(start, len(vector))]


def bind_ortvalue_input(io_binding: ort.IOBinding, name: str, value: ort.OrtValue) -> None:
    io_binding.bind_input(
        name,
        value.device_name().lower(),
        0,
        value.element_type(),
        value.shape(),
        value.data_ptr(),
    )


def ort_dtype_from_str(name: str) -> Any:
    mapping = {
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(float32)": np.float32,
        "tensor(int64)": np.int64,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported ORT tensor type: {name}") from exc


def ort_shape_to_static_list(shape: Sequence[Any], fallback_batch: int = 1) -> List[int]:
    resolved: List[int] = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif index == 0:
            resolved.append(fallback_batch)
        else:
            raise ValueError(f"Cannot allocate OrtValue for dynamic shape {shape}.")
    return resolved


def ort_shape_to_alloc_list(shape: Sequence[Any], fallback_batch: int = 1, forced_batch: int | None = None) -> List[int]:
    resolved = ort_shape_to_static_list(shape, fallback_batch=fallback_batch)
    if forced_batch is not None and resolved:
        resolved[0] = int(forced_batch)
    return resolved


def resolve_shape_with_reference(shape: Sequence[Any], reference_shape: Sequence[int] | None, fallback_batch: int = 1) -> List[int]:
    resolved: List[int] = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif index == 0:
            resolved.append(fallback_batch)
        elif reference_shape is not None and index < len(reference_shape):
            resolved.append(int(reference_shape[index]))
        else:
            raise ValueError(f"Cannot resolve dynamic shape {shape} without reference shape.")
    return resolved


def ortvalue_to_cupy(value: ort.OrtValue, dtype: Any):
    if cp is None:
        raise RuntimeError("CuPy is required for batched CUDA decode scheduling.")
    shape = tuple(int(dim) for dim in value.shape())
    nbytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    memory = cp.cuda.UnownedMemory(int(value.data_ptr()), nbytes, value)
    pointer = cp.cuda.MemoryPointer(memory, 0)
    return cp.ndarray(shape=shape, dtype=dtype, memptr=pointer)


def apply_cache_updates(cache_arrays: Sequence[Any], update_arrays: Sequence[Any], positions: np.ndarray, active_count: int) -> None:
    if cp is None:
        raise RuntimeError("CuPy is required for device-side cache updates.")
    if active_count <= 0:
        return
    row_index = cp.arange(active_count, dtype=cp.int64)
    position_index = cp.asarray(np.asarray(positions[:active_count], dtype=np.int64))
    for cache_array, update_array in zip(cache_arrays, update_arrays):
        cache_array[row_index, :, position_index, :] = update_array[:active_count, :, 0, :]
    cp.cuda.get_current_stream().synchronize()


class QueueFullError(RuntimeError):
    pass


class ServiceClosedError(RuntimeError):
    pass


@dataclass
class DecoderThreadState:
    prefill_binding: ort.IOBinding
    decode_binding: ort.IOBinding
    logits_output: ort.OrtValue
    cache_values: List[ort.OrtValue]
    update_outputs: List[ort.OrtValue]
    token_input: np.ndarray
    cache_position: np.ndarray
    cache_gpu: List[Any] = field(default_factory=list)
    update_gpu: List[Any] = field(default_factory=list)
    prefill_cache_inputs: List[ort.OrtValue] = field(default_factory=list)


@dataclass
class PreparedDecodeRequest:
    request: "ServiceRequest"
    input_ids: np.ndarray
    audio_features: np.ndarray
    audio_seconds: float
    complete: Callable[[Dict[str, Any] | None, Exception | None], None]


@dataclass
class DecodeSlot:
    prepared: PreparedDecodeRequest
    generated_token_ids: List[int]
    current_token_id: int
    cache_position: int
    inference_start: float


class OnnxAsrRuntime:
    def __init__(self, onnx_dir: Path, providers: List[str], max_new_tokens: int) -> None:
        self.onnx_dir = Path(onnx_dir).resolve()
        self.max_new_tokens = max_new_tokens
        self.metadata = json.loads((self.onnx_dir / "metadata.json").read_text(encoding="utf-8"))
        self.ort_dtype = np.float16 if self.metadata["dtype"] == "float16" else np.float32
        encoder_model = self.metadata.get("encoder_model")
        self.encoder_session: ort.InferenceSession | None = None
        self.conv_session: ort.InferenceSession | None = None
        self.audio_session: ort.InferenceSession | None = None
        if encoder_model is not None:
            self.encoder_session = load_session(self.onnx_dir / encoder_model, providers=providers)
        else:
            self.conv_session = load_session(self.onnx_dir / self.metadata["conv_embed_model"], providers=providers)
            self.audio_session = load_session(self.onnx_dir / self.metadata["audio_stack_model"], providers=providers)
        prefill_model = self.metadata.get("prefill_model", self.metadata.get("decoder_model"))
        decode_model = self.metadata.get("decode_model", self.metadata.get("decoder_model", prefill_model))
        if prefill_model is None or decode_model is None:
            raise ValueError("metadata.json is missing decoder model entries.")
        prefill_path = (self.onnx_dir / prefill_model).resolve()
        decode_path = (self.onnx_dir / decode_model).resolve()
        self.shared_decoder_session = prefill_path == decode_path
        if self.shared_decoder_session:
            decoder_session = load_session(prefill_path, providers=providers)
            self.prefill_session = decoder_session
            self.decode_session = decoder_session
        else:
            self.prefill_session = load_session(prefill_path, providers=providers)
            self.decode_session = load_session(decode_path, providers=providers)
        self.ensure_bundled_assets()
        self._chat_template: str | None = None
        chat_template_path = self.onnx_dir / "chat_template.json"
        if chat_template_path.exists():
            self._chat_template = json.loads(chat_template_path.read_text(encoding="utf-8"))["chat_template"]
        self.prefill_input_items = self.prefill_session.get_inputs()
        self.decode_input_items = self.decode_session.get_inputs()
        self.decode_output_items = self.decode_session.get_outputs()
        self.prefill_output_items = self.prefill_session.get_outputs()
        self.eos_token_ids = {int(token_id) for token_id in self.metadata["eos_token_ids"]}
        self._thread_local = threading.local()
        self._prefill_cache_specs: List[tuple[List[int], Any]] = []
        self._shared_decode_audio_features = np.zeros((1, int(self.metadata["audio_output_dim"])), dtype=self.ort_dtype)
        if len(self.prefill_input_items) >= 3 and self.prefill_input_items[2].name == "cache_position":
            for item in self.prefill_input_items[3:]:
                shape = [int(dim) for dim in item.shape]
                self._prefill_cache_specs.append((shape, ort_dtype_from_str(item.type)))
        self._decode_logits_shape = ort_shape_to_static_list(self.decode_output_items[0].shape)
        self._decode_logits_dtype = ort_dtype_from_str(self.decode_output_items[0].type)
        self._decode_cache_specs = [
            (
                resolve_shape_with_reference(
                    item.shape,
                    self._prefill_cache_specs[index][0] if index < len(self._prefill_cache_specs) else None,
                ),
                ort_dtype_from_str(item.type),
            )
            for index, item in enumerate(self.decode_output_items[1:])
        ]

    def ensure_bundled_assets(self) -> None:
        tokenizer_ready = (self.onnx_dir / "tokenizer_config.json").exists() or (self.onnx_dir / "tokenizer.json").exists()
        if not tokenizer_ready:
            raise FileNotFoundError(
                f"Missing tokenizer assets in ONNX directory: {self.onnx_dir}. Re-export the model with bundled assets."
            )
        if not (self.onnx_dir / "preprocessor_config.json").exists():
            raise FileNotFoundError(
                f"Missing required bundled asset `preprocessor_config.json` in ONNX directory: {self.onnx_dir}. Re-export the model."
            )

    def get_text_assets(self) -> tuple[Any, Any]:
        assets = getattr(self._thread_local, "text_assets", None)
        if assets is None:
            tokenizer = AutoTokenizer.from_pretrained(str(self.onnx_dir), fix_mistral_regex=True)
            if self._chat_template is not None:
                tokenizer.chat_template = self._chat_template
            feature_extractor = WhisperFeatureExtractor.from_pretrained(str(self.onnx_dir))
            assets = (tokenizer, feature_extractor)
            self._thread_local.text_assets = assets
        return assets

    def get_decoder_thread_state(self) -> DecoderThreadState:
        state = getattr(self._thread_local, "decoder_state", None)
        if state is None:
            cache_values = [
                ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, device_type="cuda", device_id=0)
                for shape, dtype in self._prefill_cache_specs
            ]
            update_outputs = [
                ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, device_type="cuda", device_id=0)
                for shape, dtype in self._decode_cache_specs
            ]
            state = DecoderThreadState(
                prefill_binding=self.prefill_session.io_binding(),
                decode_binding=self.decode_session.io_binding(),
                logits_output=ort.OrtValue.ortvalue_from_shape_and_type(
                    self._decode_logits_shape,
                    self._decode_logits_dtype,
                    device_type="cpu",
                    device_id=0,
                ),
                cache_values=cache_values,
                update_outputs=update_outputs,
                token_input=np.zeros((1, 1), dtype=np.int64),
                cache_position=np.zeros((1, 1), dtype=np.int64),
                cache_gpu=[ortvalue_to_cupy(value, dtype) for value, (_, dtype) in zip(cache_values, self._prefill_cache_specs)],
                update_gpu=[ortvalue_to_cupy(value, dtype) for value, (_, dtype) in zip(update_outputs, self._decode_cache_specs)],
                prefill_cache_inputs=[
                    ort.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), "cuda", 0)
                    for shape, dtype in self._prefill_cache_specs
                ],
            )
            self._thread_local.decoder_state = state
        return state

    def prepare_audio_features(self, waveform: np.ndarray) -> Tuple[np.ndarray, float]:
        _, feature_extractor = self.get_text_assets()
        feature_inputs = feature_extractor(
            [waveform],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="np",
        )
        input_features = np.asarray(feature_inputs["input_features"][0], dtype=np.float32)
        feature_attention_mask = np.asarray(feature_inputs["attention_mask"][0], dtype=np.int64)
        feature_len = int(feature_attention_mask.sum())

        window = int(self.metadata["n_window"]) * 2
        chunk_num = (feature_len + window - 1) // window
        padded_total = chunk_num * window
        feature_slice = input_features[:, :feature_len]
        if padded_total > feature_len:
            feature_slice = np.pad(feature_slice, ((0, 0), (0, padded_total - feature_len)), mode="constant")
        padded_feature = (
            feature_slice.T.reshape(chunk_num, window, feature_slice.shape[0]).transpose(0, 2, 1)[:, None, :, :]
        ).astype(self.ort_dtype, copy=False)

        chunk_lengths = np.full((chunk_num,), window, dtype=np.int64)
        chunk_lengths[-1] = feature_len - window * (chunk_num - 1)
        if self.encoder_session is not None:
            encoder_outputs = run_session(
                self.encoder_session,
                {
                    "padded_feature": padded_feature,
                    "chunk_lengths": chunk_lengths,
                },
            )
            audio_features = np.asarray(encoder_outputs["audio_features"], dtype=self.ort_dtype)
        else:
            conv_outputs = run_session(self.conv_session, {"padded_feature": padded_feature})
            padded_embed = np.asarray(conv_outputs["padded_embed"], dtype=self.ort_dtype)

            lengths_after = feature_output_lengths(chunk_lengths).astype(np.int64)
            mask_after = np.arange(padded_embed.shape[1])[None, :] < lengths_after[:, None]
            hidden_states = padded_embed[mask_after]

            audio_outputs = run_session(
                self.audio_session,
                {"hidden_states": hidden_states.astype(self.ort_dtype, copy=False)},
            )
            audio_features = np.asarray(audio_outputs["audio_features"], dtype=self.ort_dtype)
        return audio_features, float(waveform.shape[0]) / 16000.0

    def build_prompt(self, audio_features: np.ndarray, context: str, language: str | None) -> np.ndarray:
        tokenizer, _ = self.get_text_assets()
        base_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": context},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = base_prompt.replace(self.metadata["audio_token"], self.metadata["audio_token"] * int(audio_features.shape[0]), 1)
        if language:
            prompt = prompt + f"language {language}<asr_text>"
        tokenized = tokenizer(prompt, return_tensors="np")
        input_ids = np.asarray(tokenized["input_ids"], dtype=np.int64)
        static_cache_len = int(self.metadata["static_cache_len"])
        if input_ids.shape[1] + self.max_new_tokens > static_cache_len:
            raise ValueError(
                f"Prompt length {input_ids.shape[1]} + max_new_tokens {self.max_new_tokens} exceeds static_cache_len {static_cache_len}."
            )
        return input_ids

    def prepare_request(
        self,
        audio: tuple[np.ndarray, int],
        context: str = "",
        language: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        waveform = normalize_audio_waveform(audio[0], int(audio[1]))
        audio_features, audio_seconds = self.prepare_audio_features(waveform)
        input_ids = self.build_prompt(audio_features, context=context, language=language)
        return input_ids, audio_features, audio_seconds

    def prefill_request(self, input_ids: np.ndarray, audio_features: np.ndarray) -> Tuple[int, List[ort.OrtValue]]:
        state = self.get_decoder_thread_state()
        prefill_binding = state.prefill_binding
        prefill_binding.clear_binding_inputs()
        prefill_binding.clear_binding_outputs()
        if len(self.prefill_input_items) >= 3 and self.prefill_input_items[2].name == "cache_position":
            prefill_binding.bind_cpu_input(self.prefill_input_items[0].name, input_ids)
            prefill_binding.bind_cpu_input(self.prefill_input_items[1].name, audio_features)
            prefill_binding.bind_cpu_input(
                self.prefill_input_items[2].name,
                np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1),
            )
            for item, value in zip(self.prefill_input_items[3:], state.prefill_cache_inputs):
                bind_ortvalue_input(prefill_binding, item.name, value)
        else:
            prefill_binding.bind_cpu_input(self.prefill_input_items[0].name, input_ids)
            prefill_binding.bind_cpu_input(self.prefill_input_items[1].name, audio_features)
        prefill_binding.bind_ortvalue_output(self.prefill_output_items[0].name, state.logits_output)
        for item, value in zip(self.prefill_output_items[1:], state.cache_values):
            prefill_binding.bind_ortvalue_output(item.name, value)
        self.prefill_session.run_with_iobinding(prefill_binding)
        logits = state.logits_output.numpy()
        return int(np.argmax(logits[0])), state.cache_values

    def decode_tokens(self, input_ids: np.ndarray, audio_features: np.ndarray) -> Tuple[List[int], float]:
        inference_start = time.perf_counter()
        next_token_id, _ = self.prefill_request(input_ids, audio_features)
        state = self.get_decoder_thread_state()

        generated_token_ids: List[int] = []
        state.cache_position[0, 0] = input_ids.shape[1]
        for _ in range(self.max_new_tokens):
            if next_token_id in self.eos_token_ids:
                break
            generated_token_ids.append(next_token_id)
            decode_binding = state.decode_binding
            decode_binding.clear_binding_inputs()
            decode_binding.clear_binding_outputs()
            state.token_input[0, 0] = next_token_id
            decode_binding.bind_cpu_input(self.decode_input_items[0].name, state.token_input)
            if self.shared_decoder_session:
                decode_binding.bind_cpu_input(self.decode_input_items[1].name, self._shared_decode_audio_features)
                decode_binding.bind_cpu_input(self.decode_input_items[2].name, state.cache_position)
                cache_inputs = self.decode_input_items[3:]
            else:
                decode_binding.bind_cpu_input(self.decode_input_items[1].name, state.cache_position)
                cache_inputs = self.decode_input_items[2:]
            for item, value in zip(cache_inputs, state.cache_values):
                bind_ortvalue_input(decode_binding, item.name, value)
            decode_binding.bind_ortvalue_output(self.decode_output_items[0].name, state.logits_output)
            for item, value in zip(self.decode_output_items[1:], state.update_outputs):
                decode_binding.bind_ortvalue_output(item.name, value)
            self.decode_session.run_with_iobinding(decode_binding)
            apply_cache_updates(state.cache_gpu, state.update_gpu, state.cache_position[:, 0], active_count=1)
            logits = state.logits_output.numpy()
            next_token_id = int(np.argmax(logits[0]))
            state.cache_position[0, 0] += 1

        inference_seconds = time.perf_counter() - inference_start
        return generated_token_ids, inference_seconds

    def decode_generated_tokens(self, generated_token_ids: List[int], language: str | None, audio_seconds: float, inference_seconds: float) -> Dict[str, Any]:
        tokenizer, _ = self.get_text_assets()
        raw = tokenizer.batch_decode(
            [generated_token_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        output_language, text = parse_asr_output(raw, user_language=language)
        return {
            "language": output_language,
            "text": text,
            "raw": raw,
            "audio_seconds": audio_seconds,
            "inference_seconds": inference_seconds,
            "num_generated_tokens": len(generated_token_ids),
        }

    def transcribe_waveform(
        self,
        audio: tuple[np.ndarray, int],
        context: str = "",
        language: str | None = None,
    ) -> Dict[str, Any]:
        input_ids, audio_features, audio_seconds = self.prepare_request(audio, context=context, language=language)
        generated_token_ids, inference_seconds = self.decode_tokens(input_ids, audio_features)
        return self.decode_generated_tokens(generated_token_ids, language=language, audio_seconds=audio_seconds, inference_seconds=inference_seconds)

    def transcribe_input(self, audio_input: str, context: str = "", language: str | None = None) -> Dict[str, Any]:
        audio, sample_rate = load_audio_any(audio_input)
        return self.transcribe_waveform((audio, sample_rate), context=context, language=language)


@dataclass
class ServiceRequest:
    request_id: int
    kind: str
    payload: Any
    context: str
    language: str | None
    audio_seconds_hint: float | None
    future: Future[Dict[str, Any]]


class BatchedDecodeScheduler:
    def __init__(self, runtime: OnnxAsrRuntime, batch_size: int, max_queue_size: int = 128) -> None:
        if cp is None:
            raise RuntimeError("Batched CUDA decode requires CuPy in the runtime environment.")
        self.runtime = runtime
        self.batch_size = max(1, batch_size)
        self.max_queue_size = max(1, max_queue_size)
        self._queue: queue.Queue[PreparedDecodeRequest | None] = queue.Queue(self.max_queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"onnx-batch-decode-{id(self):x}", daemon=False)

        self.decode_binding = self.runtime.decode_session.io_binding()
        self.decode_input_items = self.runtime.decode_input_items
        self.decode_output_items = self.runtime.decode_output_items
        self.current_bank = 0
        self.active_slots: List[DecodeSlot] = []

        self.token_input = np.zeros((self.batch_size, 1), dtype=np.int64)
        self.cache_position = np.zeros((self.batch_size, 1), dtype=np.int64)
        self.logits_output = ort.OrtValue.ortvalue_from_shape_and_type(
            ort_shape_to_alloc_list(self.decode_output_items[0].shape, fallback_batch=self.batch_size, forced_batch=self.batch_size),
            ort_dtype_from_str(self.decode_output_items[0].type),
            device_type="cuda",
            device_id=0,
        )
        self.logits_gpu = ortvalue_to_cupy(self.logits_output, ort_dtype_from_str(self.decode_output_items[0].type))
        self.cache_specs = [
            (
                ort_shape_to_alloc_list(shape, fallback_batch=self.batch_size, forced_batch=self.batch_size),
                dtype,
            )
            for shape, dtype in self.runtime._decode_cache_specs
        ]
        self.cache_banks: List[List[ort.OrtValue]] = [
            [
                ort.OrtValue.ortvalue_from_shape_and_type(shape, dtype, device_type="cuda", device_id=0)
                for shape, dtype in self.cache_specs
            ]
            for _ in range(2)
        ]
        self.cache_banks_gpu = [
            [ortvalue_to_cupy(value, dtype) for value, (_, dtype) in zip(bank, self.cache_specs)]
            for bank in self.cache_banks
        ]
        self._thread.start()

    def submit(self, prepared: PreparedDecodeRequest) -> None:
        self._queue.put(prepared)

    def pending_size(self) -> int:
        return self._queue.qsize() + len(self.active_slots)

    def _copy_prefill_cache_to_slot(self, slot_index: int, cache_values: List[ort.OrtValue]) -> None:
        dst_bank = self.cache_banks_gpu[self.current_bank]
        for dst, src_value, (_, dtype) in zip(dst_bank, cache_values, self.cache_specs):
            src = ortvalue_to_cupy(src_value, dtype)
            cp.copyto(dst[slot_index : slot_index + 1], src)
        cp.cuda.get_current_stream().synchronize()

    def _finalize_slot(self, slot: DecodeSlot, success: bool, error: Exception | None = None) -> None:
        if success:
            result = self.runtime.decode_generated_tokens(
                slot.generated_token_ids,
                language=slot.prepared.request.language,
                audio_seconds=slot.prepared.audio_seconds,
                inference_seconds=time.perf_counter() - slot.inference_start,
            )
            slot.prepared.complete(result, None)
        else:
            slot.prepared.complete(None, error or RuntimeError("Batched decode failed."))

    def _admit_request(self, prepared: PreparedDecodeRequest) -> None:
        if prepared.request.future.cancelled():
            prepared.complete(None, RuntimeError("Request was cancelled before decode admission."))
            return
        inference_start = time.perf_counter()
        first_token_id, cache_values = self.runtime.prefill_request(prepared.input_ids, prepared.audio_features)
        if first_token_id in self.runtime.eos_token_ids:
            prepared.complete(
                self.runtime.decode_generated_tokens([], prepared.request.language, prepared.audio_seconds, time.perf_counter() - inference_start),
                None,
            )
            return

        generated_token_ids = [first_token_id]
        if len(generated_token_ids) >= self.runtime.max_new_tokens:
            prepared.complete(
                self.runtime.decode_generated_tokens(
                    generated_token_ids,
                    prepared.request.language,
                    prepared.audio_seconds,
                    time.perf_counter() - inference_start,
                ),
                None,
            )
            return

        slot_index = len(self.active_slots)
        self._copy_prefill_cache_to_slot(slot_index, cache_values)
        self.active_slots.append(
            DecodeSlot(
                prepared=prepared,
                generated_token_ids=generated_token_ids,
                current_token_id=first_token_id,
                cache_position=int(prepared.input_ids.shape[1]),
                inference_start=inference_start,
            )
        )

    def _drain_new_requests(self, block: bool) -> None:
        timeout = 0.05 if block else 0.0
        while len(self.active_slots) < self.batch_size:
            try:
                prepared = self._queue.get(timeout=timeout) if block else self._queue.get_nowait()
            except queue.Empty:
                break
            if prepared is None:
                self._queue.task_done()
                self._stop_event.set()
                break
            try:
                self._admit_request(prepared)
            except Exception as exc:
                prepared.complete(None, exc)
            finally:
                self._queue.task_done()
            block = False
            timeout = 0.0

    def _decode_step(self) -> None:
        active_count = len(self.active_slots)
        if active_count == 0:
            return

        self.token_input.fill(0)
        self.cache_position.fill(0)
        for slot_index, slot in enumerate(self.active_slots):
            self.token_input[slot_index, 0] = slot.current_token_id
            self.cache_position[slot_index, 0] = slot.cache_position

        self.decode_binding.clear_binding_inputs()
        self.decode_binding.clear_binding_outputs()
        self.decode_binding.bind_cpu_input(self.decode_input_items[0].name, self.token_input)
        self.decode_binding.bind_cpu_input(self.decode_input_items[1].name, self.cache_position)
        for item, value in zip(self.decode_input_items[2:], self.cache_banks[self.current_bank]):
            bind_ortvalue_input(self.decode_binding, item.name, value)

        self.decode_binding.bind_ortvalue_output(self.decode_output_items[0].name, self.logits_output)
        next_bank = 1 - self.current_bank
        for item, value in zip(self.decode_output_items[1:], self.cache_banks[next_bank]):
            self.decode_binding.bind_ortvalue_output(item.name, value)

        self.runtime.decode_session.run_with_iobinding(self.decode_binding)
        self.decode_binding.synchronize_outputs()
        next_token_ids = cp.asnumpy(cp.argmax(self.logits_gpu[:active_count], axis=1)).astype(np.int64)
        self.current_bank = next_bank
        survivors: List[tuple[int, DecodeSlot]] = []
        for slot_index, slot in enumerate(self.active_slots):
            predicted_token_id = int(next_token_ids[slot_index])
            slot.cache_position += 1
            if predicted_token_id in self.runtime.eos_token_ids:
                self._finalize_slot(slot, success=True)
                continue
            slot.generated_token_ids.append(predicted_token_id)
            if len(slot.generated_token_ids) >= self.runtime.max_new_tokens:
                self._finalize_slot(slot, success=True)
                continue
            slot.current_token_id = predicted_token_id
            survivors.append((slot_index, slot))

        if not survivors:
            self.active_slots = []
            return

        moved = False
        for new_index, (old_index, slot) in enumerate(survivors):
            if new_index != old_index:
                for dst in self.cache_banks_gpu[self.current_bank]:
                    cp.copyto(dst[new_index : new_index + 1], dst[old_index : old_index + 1])
                moved = True
        if moved:
            cp.cuda.get_current_stream().synchronize()
        self.active_slots = [slot for _, slot in survivors]

    def _run(self) -> None:
        while True:
            self._drain_new_requests(block=not self.active_slots)
            if self.active_slots:
                self._decode_step()
                continue
            if self._stop_event.is_set() and self._queue.empty():
                break

    def close(self, wait: bool = True) -> None:
        self._stop_event.set()
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                if not wait:
                    break
                time.sleep(0.01)
        if wait:
            self._thread.join()


class OnnxAsrService:
    def __init__(
        self,
        onnx_dir: Path,
        providers: List[str],
        max_new_tokens: int,
        workers: int = 4,
        replicas: int = 1,
        max_inflight: int | None = None,
        max_queue_size: int = 128,
        short_audio_seconds: float = 20.0,
    ) -> None:
        self.replicas = max(1, replicas)
        self.runtimes = [OnnxAsrRuntime(onnx_dir=onnx_dir, providers=providers, max_new_tokens=max_new_tokens) for _ in range(self.replicas)]
        self.runtime = self.runtimes[0]
        self.workers = max(1, workers)
        self.max_inflight = max(1, max_inflight or workers)
        self.max_queue_size = max(1, max_queue_size)
        self.short_audio_seconds = short_audio_seconds
        self.decode_batch_size = max(1, math.ceil(self.max_inflight / self.replicas))
        self._batched_decode_enabled = (
            self.decode_batch_size > 1
            and
            cp is not None
            and all("CUDAExecutionProvider" in runtime.decode_session.get_providers() for runtime in self.runtimes)
            and self.runtime.metadata.get("decode_batch_mode") == "dynamic"
        )
        self._sequence = itertools.count()
        self._request_ids = itertools.count(1)
        self._queue: queue.PriorityQueue[tuple[int, int, ServiceRequest | None]] = queue.PriorityQueue(self.max_queue_size)
        self._inflight_semaphore = threading.BoundedSemaphore(self.max_inflight)
        self._stop_event = threading.Event()
        self._stats_lock = threading.Lock()
        self._active = 0
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._threads: List[threading.Thread] = []
        self._scheduler_rr = itertools.count()
        self.schedulers: List[BatchedDecodeScheduler] = []
        if self._batched_decode_enabled:
            self.schedulers = [
                BatchedDecodeScheduler(
                    runtime=runtime,
                    batch_size=self.decode_batch_size,
                    max_queue_size=max(self.max_queue_size, self.decode_batch_size * 4),
                )
                for runtime in self.runtimes
            ]
        for worker_idx in range(self.workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"onnx-asr-worker-{worker_idx:02d}",
                daemon=False,
            )
            thread.start()
            self._threads.append(thread)

    def _priority_for(self, audio_seconds_hint: float | None) -> int:
        if audio_seconds_hint is None:
            return 0
        return 0 if audio_seconds_hint <= self.short_audio_seconds else 1

    def _submit(self, kind: str, payload: Any, context: str, language: str | None, audio_seconds_hint: float | None) -> Future[Dict[str, Any]]:
        future: Future[Dict[str, Any]] = Future()
        if self._stop_event.is_set():
            future.set_exception(ServiceClosedError("Service has been closed."))
            return future
        request = ServiceRequest(
            request_id=next(self._request_ids),
            kind=kind,
            payload=payload,
            context=context,
            language=language,
            audio_seconds_hint=audio_seconds_hint,
            future=future,
        )
        try:
            self._queue.put_nowait((self._priority_for(audio_seconds_hint), next(self._sequence), request))
        except queue.Full:
            future.set_exception(QueueFullError(f"Service queue is full (max_queue_size={self.max_queue_size})."))
            return future
        with self._stats_lock:
            self._submitted += 1
            self._active += 1
        return future

    def _finish_future(self, future: Future[Dict[str, Any]], result: Dict[str, Any] | None, exc: Exception | None) -> None:
        try:
            if exc is None:
                if not future.cancelled():
                    future.set_result(result if result is not None else {})
                with self._stats_lock:
                    self._completed += 1
            else:
                if not future.cancelled():
                    future.set_exception(exc)
                with self._stats_lock:
                    self._failed += 1
        finally:
            with self._stats_lock:
                self._active -= 1

    def _prepare_request(self, runtime: OnnxAsrRuntime, request: ServiceRequest) -> PreparedDecodeRequest:
        if request.kind == "waveform":
            audio = request.payload
        elif request.kind == "input":
            audio = load_audio_any(request.payload)
        else:
            raise ValueError(f"Unsupported request kind: {request.kind}")
        input_ids, audio_features, audio_seconds = runtime.prepare_request(audio, context=request.context, language=request.language)
        return PreparedDecodeRequest(
            request=request,
            input_ids=input_ids,
            audio_features=audio_features,
            audio_seconds=audio_seconds,
            complete=lambda result, exc: self._finish_future(request.future, result, exc),
        )

    def _select_scheduler(self) -> BatchedDecodeScheduler:
        scheduler_index = next(self._scheduler_rr) % len(self.schedulers)
        best_index = scheduler_index
        best_depth = self.schedulers[best_index].pending_size()
        for offset in range(1, len(self.schedulers)):
            candidate_index = (scheduler_index + offset) % len(self.schedulers)
            candidate_depth = self.schedulers[candidate_index].pending_size()
            if candidate_depth < best_depth:
                best_index = candidate_index
                best_depth = candidate_depth
        return self.schedulers[best_index]

    def submit_waveform(
        self,
        audio: tuple[np.ndarray, int],
        context: str = "",
        language: str | None = None,
        audio_seconds_hint: float | None = None,
    ) -> Future[Dict[str, Any]]:
        if audio_seconds_hint is None:
            audio_seconds_hint = float(len(audio[0])) / float(audio[1]) if int(audio[1]) > 0 else None
        return self._submit("waveform", audio, context, language, audio_seconds_hint)

    def submit_input(
        self,
        audio_input: str,
        context: str = "",
        language: str | None = None,
        audio_seconds_hint: float | None = None,
    ) -> Future[Dict[str, Any]]:
        return self._submit("input", audio_input, context, language, audio_seconds_hint)

    def transcribe_waveform(
        self,
        audio: tuple[np.ndarray, int],
        context: str = "",
        language: str | None = None,
        audio_seconds_hint: float | None = None,
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        return self.submit_waveform(audio, context=context, language=language, audio_seconds_hint=audio_seconds_hint).result(timeout=timeout)

    def transcribe_input(
        self,
        audio_input: str,
        context: str = "",
        language: str | None = None,
        audio_seconds_hint: float | None = None,
        timeout: float | None = None,
    ) -> Dict[str, Any]:
        return self.submit_input(audio_input, context=context, language=language, audio_seconds_hint=audio_seconds_hint).result(timeout=timeout)

    def _worker_loop(self) -> None:
        while True:
            try:
                _, _, request = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            if request is None:
                self._queue.task_done()
                break
            if request.future.cancelled():
                self._finish_future(request.future, None, RuntimeError("Request was cancelled."))
                self._queue.task_done()
                continue
            runtime = self.runtimes[request.request_id % self.replicas]
            try:
                if self._batched_decode_enabled:
                    prepared = self._prepare_request(runtime, request)
                    self._select_scheduler().submit(prepared)
                else:
                    if request.kind == "waveform":
                        result = runtime.transcribe_waveform(request.payload, context=request.context, language=request.language)
                    elif request.kind == "input":
                        result = runtime.transcribe_input(request.payload, context=request.context, language=request.language)
                    else:
                        raise ValueError(f"Unsupported request kind: {request.kind}")
                    self._finish_future(request.future, result, None)
            except Exception as exc:
                self._finish_future(request.future, None, exc)
            finally:
                self._queue.task_done()

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                "replicas": self.replicas,
                "workers": self.workers,
                "max_inflight": self.max_inflight,
                "max_queue_size": self.max_queue_size,
                "short_audio_seconds": self.short_audio_seconds,
                "decode_batch_size": self.decode_batch_size,
                "batched_decode_enabled": self._batched_decode_enabled,
                "queue_depth": self._queue.qsize(),
                "active_requests": self._active,
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
            }

    def close(self, wait: bool = True) -> None:
        self._stop_event.set()
        for _ in self._threads:
            while True:
                try:
                    self._queue.put((2, next(self._sequence), None), timeout=0.1)
                    break
                except queue.Full:
                    continue
        if wait:
            for thread in self._threads:
                thread.join()
            for scheduler in self.schedulers:
                scheduler.close(wait=True)
        else:
            for scheduler in self.schedulers:
                scheduler.close(wait=False)
        self.schedulers = []
        self.runtimes = []
        self.runtime = None  # type: ignore[assignment]

    def __enter__(self) -> "OnnxAsrService":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(wait=True)
