#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from onnx_asr_service import OnnxAsrService, QueueFullError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Qwen3-ASR ONNX as a local HTTP service.")
    parser.add_argument("--onnx-dir", required=True, help="Directory containing exported ONNX files and metadata.json.")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP listen host.")
    parser.add_argument("--port", type=int, default=18080, help="HTTP listen port.")
    parser.add_argument("--workers", type=int, default=4, help="Number of service worker threads.")
    parser.add_argument("--max-inflight", type=int, default=4, help="Maximum in-flight inference requests.")
    parser.add_argument("--max-queue-size", type=int, default=64, help="Maximum queued requests.")
    parser.add_argument("--short-audio-seconds", type=float, default=20.0, help="Short-audio priority threshold.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation cap.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime providers.",
    )
    return parser.parse_args()


class AsrRequestHandler(BaseHTTPRequestHandler):
    service: OnnxAsrService

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        payload = {
            "status": "ok",
            "model_name": self.service.runtime.metadata["model_name"],
            "service": self.service.stats(),
        }
        self.send_json(HTTPStatus.OK, payload)

    def do_POST(self) -> None:
        if self.path != "/transcribe":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            request_bytes = self.rfile.read(content_length)
            payload = json.loads(request_bytes.decode("utf-8"))
        except Exception as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid_json: {exc}"})
            return

        audio = payload.get("audio")
        if not isinstance(audio, str) or not audio:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "`audio` is required and must be a string."})
            return

        context = payload.get("context", "")
        language = payload.get("language")
        audio_seconds_hint = payload.get("audio_seconds_hint")
        timeout_seconds = float(payload.get("timeout_seconds", 600.0))
        print_raw = bool(payload.get("print_raw", False))

        try:
            future = self.service.submit_input(
                audio_input=audio,
                context=context,
                language=language,
                audio_seconds_hint=audio_seconds_hint,
            )
            result = future.result(timeout=timeout_seconds)
        except QueueFullError as exc:
            self.send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": str(exc), "service": self.service.stats()})
            return
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        response = {
            "language": result["language"],
            "text": result["text"],
            "num_generated_tokens": result["num_generated_tokens"],
            "elapsed_seconds": result["inference_seconds"],
            "audio_seconds": result["audio_seconds"],
            "rtf": result["inference_seconds"] / result["audio_seconds"] if result["audio_seconds"] else 0.0,
        }
        if print_raw:
            response["raw"] = result["raw"]
        self.send_json(HTTPStatus.OK, response)


def main() -> None:
    args = parse_args()
    service = OnnxAsrService(
        onnx_dir=Path(args.onnx_dir).resolve(),
        providers=args.providers,
        max_new_tokens=args.max_new_tokens,
        workers=args.workers,
        max_inflight=args.max_inflight,
        max_queue_size=args.max_queue_size,
        short_audio_seconds=args.short_audio_seconds,
    )
    AsrRequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), AsrRequestHandler)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "model_name": service.runtime.metadata["model_name"],
                "service": service.stats(),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close(wait=True)


if __name__ == "__main__":
    main()
