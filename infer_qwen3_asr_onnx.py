#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from onnx_asr_service import OnnxAsrRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-audio Qwen3-ASR inference with ONNX Runtime only, without PyTorch."
    )
    parser.add_argument("--model", required=True, help="Local model directory for tokenizer and feature extractor.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing exported ONNX files and metadata.json.")
    parser.add_argument("--audio", required=True, help="Audio path, URL or base64 data URL.")
    parser.add_argument("--context", default="", help="Optional system context.")
    parser.add_argument("--language", default=None, help="Optional forced language.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation cap.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime providers.",
    )
    parser.add_argument("--print-raw", action="store_true", help="Print raw model output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = OnnxAsrRuntime(
        model_dir=Path(args.model).resolve(),
        onnx_dir=Path(args.onnx_dir).resolve(),
        providers=args.providers,
        max_new_tokens=args.max_new_tokens,
    )
    result = runtime.transcribe_input(args.audio, context=args.context, language=args.language)
    payload = {
        "language": result["language"],
        "text": result["text"],
        "num_generated_tokens": result["num_generated_tokens"],
        "elapsed_seconds": result["inference_seconds"],
        "audio_seconds": result["audio_seconds"],
        "rtf": result["inference_seconds"] / result["audio_seconds"] if result["audio_seconds"] else 0.0,
    }
    if args.print_raw:
        payload["raw"] = result["raw"]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
