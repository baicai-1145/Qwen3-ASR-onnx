# Qwen3-ASR-onnx

<p align="center">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/logo.png" width="360" alt="Qwen3-ASR logo"/>
</p>

<p align="center">
  面向生产部署的 Qwen3-ASR ONNX Runtime 方案
</p>

<p align="center">
  纯 ONNX Runtime 推理 · 本地模型部署 · 服务化并发 · 中英基准测试
</p>

<p align="center">
  <a href="https://modelscope.cn/collections/Qwen/Qwen3-ASR">ModelScope</a>
  ·
  <a href="https://huggingface.co/collections/Qwen/qwen3-asr">Hugging Face</a>
  ·
  <a href="https://qwen.ai/blog?id=qwen3asr">Blog</a>
  ·
  <a href="https://arxiv.org/abs/2601.21337">Paper</a>
</p>

<p align="center">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/qwen3_asr_introduction.png" width="92%" alt="Qwen3-ASR introduction"/>
</p>

## 简介

这个仓库提供一套围绕 Qwen3-ASR 的 ONNX 化落地方案，目标很明确：

- 推理阶段尽量不再依赖 PyTorch
- 保持与原模型一致的识别质量，不做质量妥协
- 支持本地单机部署、HTTP 服务化、多并发任务调度
- 已在英文 `LibriSpeech clean | other` 与中文 `WenetSpeech net | meeting` 上完成基准验证

如果你想把 Qwen3-ASR 部署成一个更轻、更易分发、跨平台更友好的 ONNX Runtime 服务，这个仓库就是为这个场景准备的。

<p align="center">
  <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/overview.jpg" width="100%" alt="Qwen3-ASR overview"/>
</p>

## 核心能力

- 纯 ONNX Runtime 推理路径，运行时默认不依赖 PyTorch
- 支持本地音频、URL、base64 音频输入
- 支持单次推理与 HTTP 服务部署
- 支持固定 KV cache 的 decoder 图与服务化并发调度
- 保留 ONNX 导出链路，便于重新导出不同配置

## 快速开始

### 1. 安装

纯 ONNX 推理最小安装：

```bash
pip install -e .
```

如果需要 ONNX 导出：

```bash
pip install -e ".[export]"
```

如果需要完整开发环境：

```bash
pip install -e ".[dev-full]"
```

### 2. 下载 ONNX 模型

可以直接从 Hugging Face 或 ModelScope 下载已经导出的 ONNX 包：

Hugging Face:

```bash
pip install -U huggingface_hub
huggingface-cli download baicai1145/Qwen3-ASR-0.6B-ONNX --local-dir ./onnx/Qwen3-ASR-0.6B
huggingface-cli download baicai1145/Qwen3-ASR-1.7B-ONNX --local-dir ./onnx/Qwen3-ASR-1.7B
```

ModelScope:

```bash
pip install -U modelscope
modelscope download --model baicai1145/Qwen3-ASR-0.6B-ONNX --local_dir ./onnx/Qwen3-ASR-0.6B
modelscope download --model baicai1145/Qwen3-ASR-1.7B-ONNX --local_dir ./onnx/Qwen3-ASR-1.7B
```

### 3. 导出 ONNX

如果你直接下载上面的 ONNX 包，可以跳过这一步。

注意：导出阶段仍然需要 PyTorch；纯 ONNX 仅针对运行时。
导出完成后，`onnx` 目录会同时打包 tokenizer、feature extractor 和 chat template，部署时默认不再需要原始模型目录。

```bash
python export_qwen3_asr_onnx.py \
  --model models/Qwen3-ASR-0.6B \
  --output-dir onnx/Qwen3-ASR-0.6B \
  --device cuda:0 \
  --dtype float16 \
  --static-cache-len 1664
```

### 4. 单文件推理

```bash
python infer_qwen3_asr_onnx.py \
  --onnx-dir onnx/Qwen3-ASR-0.6B \
  --audio samples/asr_zh.wav
```

### 5. 启动本地 HTTP 服务

```bash
python serve_qwen3_asr_onnx.py \
  --onnx-dir onnx/Qwen3-ASR-0.6B \
  --host 0.0.0.0 \
  --port 18080 \
  --workers 8 \
  --max-inflight 8
```

健康检查：

```bash
curl http://127.0.0.1:18080/health
```

推理请求：

```bash
curl -X POST http://127.0.0.1:18080/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "samples/asr_en.wav",
    "print_raw": false
  }'
```

## 依赖说明

默认安装只覆盖纯 ONNX 推理所需依赖：

- `numpy`
- `onnxruntime-gpu`
- `librosa`
- `soundfile`
- `transformers`

按需扩展：

- `export`: ONNX 导出
- `torch-baseline`: 原版 PyTorch 路径
- `vllm`: vLLM 方案
- `ui`: Gradio 相关能力
- `dev-full`: 当前仓库全量开发环境

## 主要脚本

| 脚本 | 作用 |
|---|---|
| `export_qwen3_asr_onnx.py` | 导出 ONNX 图与元数据 |
| `onnx_asr_service.py` | ONNX Runtime 推理核心与服务调度 |
| `infer_qwen3_asr_onnx.py` | 单次推理 |
| `serve_qwen3_asr_onnx.py` | 本地 HTTP 服务 |

## 说明

- 运行时目标是纯 ONNX Runtime，不依赖 PyTorch。
- 导出阶段仍需 PyTorch，因为原始模型权重与计算图来自 PyTorch 生态。
- 当前方案已经过中英文公开测试集验证，用于检查识别质量与部署可用性。
- 当前仓库更偏向部署、验证与性能工程，而不是上游训练仓库。
