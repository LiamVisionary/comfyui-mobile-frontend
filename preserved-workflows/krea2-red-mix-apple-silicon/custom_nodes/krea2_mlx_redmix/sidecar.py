from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


KREA2_DIR = Path(os.environ.get("KREA2_MLX_DIR", str(Path.home() / "comfy/krea2_alis_mlx_redmix")))
MFLUX_SITE = Path(os.environ.get(
    "KREA2_MFLUX_SITE_PACKAGES",
    str(Path.home() / "comfy/mflux-biglove/.venv/lib/python3.11/site-packages"),
))
TRANSFORMER_PATH = Path(os.environ.get(
    "KREA2_REDMIX_TRANSFORMER",
    str(KREA2_DIR / "redmix_mxfp8_fused.safetensors"),
))
OUTPUT_DIR = Path(os.environ.get(
    "KREA2_MLX_SIDECAR_OUTPUT_DIR",
    str(Path.home() / ".comfy-private.noindex/temp/krea2_mlx_sidecar"),
))
PORT = int(os.environ.get("KREA2_MLX_SIDECAR_PORT", "8796"))

for path in (str(KREA2_DIR), str(MFLUX_SITE)):
    if path not in sys.path:
        sys.path.insert(0, path)


_PIPE = None
_PIPE_KEY = None
_PIPE_LOCK = threading.Lock()


def _prepare_mlx_runtime(clear_cache=False):
    import mlx.core as mx

    info = mx.device_info()
    mx.set_wired_limit(int(info["max_recommended_working_set_size"] * 0.9))
    mx.set_cache_limit(int(os.environ.get("KREA2_MLX_CACHE_LIMIT_GB", "8")) * 1024**3)
    if clear_cache:
        mx.clear_cache()


def _pipeline():
    global _PIPE, _PIPE_KEY
    key = (str(TRANSFORMER_PATH), "mxfp8-fused", os.environ.get("KREA2_BASE_DIR"))
    with _PIPE_LOCK:
        if _PIPE is not None and _PIPE_KEY == key:
            return _PIPE
        if not TRANSFORMER_PATH.exists():
            raise FileNotFoundError(f"Krea2 Red Mix MXFP8 transformer not found: {TRANSFORMER_PATH}")
        _prepare_mlx_runtime(clear_cache=True)
        from krea2.pipeline import Krea2Pipeline

        _PIPE = Krea2Pipeline(
            transformer_path=str(TRANSFORMER_PATH),
            precision="mxfp8-fused",
            base_dir=os.environ.get("KREA2_BASE_DIR"),
        )
        _PIPE_KEY = key
        return _PIPE


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    server_version = "Krea2MLXSidecar/1.0"

    def log_message(self, fmt, *args):
        if os.environ.get("KREA2_MLX_SIDECAR_ACCESS_LOG", "0").lower() in {"1", "true", "yes", "on"}:
            print(f"[Krea2MLXSidecar] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            _json_response(self, 200, {"ok": True, "loaded": _PIPE is not None})
            return
        _json_response(self, 404, {"error": "not found"})

    def do_HEAD(self):
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/generate":
            _json_response(self, 404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            prompt = str(payload["prompt"]).strip()
            width = int(payload["width"])
            height = int(payload["height"])
            steps = int(payload["steps"])
            seed = int(payload["seed"])
            num_images = int(payload.get("num_images", 1))
            prefix = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(payload.get("prefix") or "krea2"))

            timings = {}
            start = time.perf_counter()
            pipe = _pipeline()
            timings["pipeline"] = time.perf_counter() - start

            gen_start = time.perf_counter()
            images = pipe.generate(prompt, width=width, height=height, steps=steps, seed=seed, num_images=num_images)
            timings["generate"] = time.perf_counter() - gen_start

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            paths = []
            save_start = time.perf_counter()
            run_id = uuid.uuid4().hex[:10]
            for i, image in enumerate(images):
                path = OUTPUT_DIR / f"{prefix}_{run_id}_{i:02d}.png"
                image.save(path)
                paths.append(str(path))
            timings["save"] = time.perf_counter() - save_start
            timings["total"] = time.perf_counter() - start
            print(f"[Krea2MLXSidecar] generated steps={steps} size={width}x{height} timings={timings}", flush=True)
            _json_response(self, 200, {"images": paths, "timings": timings})
        except Exception as exc:
            traceback.print_exc()
            _json_response(self, 500, {"error": str(exc), "traceback": traceback.format_exc()})


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[Krea2MLXSidecar] listening on 127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
