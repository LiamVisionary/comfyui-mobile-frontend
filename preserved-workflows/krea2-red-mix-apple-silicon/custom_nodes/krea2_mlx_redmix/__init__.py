import gc
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch

import comfy.utils


KREA2_DIR = Path(os.environ.get("KREA2_MLX_DIR", str(Path.home() / "comfy/krea2_alis_mlx_redmix")))
MFLUX_SITE = Path(os.environ.get(
    "KREA2_MFLUX_SITE_PACKAGES",
    str(Path.home() / "comfy/mflux-biglove/.venv/lib/python3.11/site-packages"),
))
TRANSFORMER_PATH = Path(os.environ.get(
    "KREA2_REDMIX_TRANSFORMER",
    str(KREA2_DIR / "redmix_mxfp8_fused.safetensors"),
))

for path in (str(KREA2_DIR), str(MFLUX_SITE)):
    if path not in sys.path:
        sys.path.insert(0, path)


_PIPE = None
_PIPE_KEY = None
_PIPE_LOCK = threading.Lock()
_SIDECAR_PROC = None
_SIDECAR_LOCK = threading.Lock()
_SIDECAR_PORT = int(os.environ.get("KREA2_MLX_SIDECAR_PORT", "8796"))
_SIDECAR_URL = f"http://127.0.0.1:{_SIDECAR_PORT}"
_SIDECAR_LOG = Path(os.environ.get(
    "KREA2_MLX_SIDECAR_LOG",
    str(Path.home() / ".comfy-private.noindex/krea2-mlx-sidecar.log"),
))
_SIDECAR_OUTPUT_DIR = Path(os.environ.get(
    "KREA2_MLX_SIDECAR_OUTPUT_DIR",
    str(Path.home() / ".comfy-private.noindex/temp/krea2_mlx_sidecar"),
))


def _prepare_mlx_runtime(clear_cache=False):
    try:
        import mlx.core as mx
        info = mx.device_info()
        mx.set_wired_limit(int(info["max_recommended_working_set_size"] * 0.9))
        mx.set_cache_limit(int(os.environ.get("KREA2_MLX_CACHE_LIMIT_GB", "8")) * 1024**3)
        if clear_cache:
            mx.clear_cache()
    except Exception:
        pass


def _clear_cache_before_each_run():
    return os.environ.get("KREA2_MLX_CLEAR_CACHE_PER_RUN", "0").lower() in {"1", "true", "yes", "on"}


def _timings_enabled():
    return os.environ.get("KREA2_MLX_TIMINGS", "0").lower() in {"1", "true", "yes", "on"}


def _request_json(path, payload=None, timeout=10):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{_SIDECAR_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _sidecar_healthy():
    try:
        return bool(_request_json("/health", timeout=1).get("ok"))
    except Exception:
        return False


def _ensure_sidecar():
    global _SIDECAR_PROC
    if _sidecar_healthy():
        return
    with _SIDECAR_LOCK:
        if _sidecar_healthy():
            return
        _SIDECAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        _SIDECAR_LOG.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "KREA2_MLX_DIR": str(KREA2_DIR),
            "KREA2_MFLUX_SITE_PACKAGES": str(MFLUX_SITE),
            "KREA2_REDMIX_TRANSFORMER": str(TRANSFORMER_PATH),
            "KREA2_MLX_SIDECAR_PORT": str(_SIDECAR_PORT),
            "KREA2_MLX_SIDECAR_OUTPUT_DIR": str(_SIDECAR_OUTPUT_DIR),
            "MLX_METAL_FAST_SYNCH": os.environ.get("MLX_METAL_FAST_SYNCH", "1"),
            "KREA2_MLX_COMPILE_FORWARD": os.environ.get("KREA2_MLX_COMPILE_FORWARD", "1"),
        })
        log = open(_SIDECAR_LOG, "ab", buffering=0)
        _SIDECAR_PROC = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("sidecar.py"))],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            close_fds=True,
        )
        try:
            subprocess.run(["/usr/sbin/taskpolicy", "-B", "-p", str(_SIDECAR_PROC.pid)], check=False)
        except Exception:
            pass
        deadline = time.time() + 20
        while time.time() < deadline:
            if _sidecar_healthy():
                return
            if _SIDECAR_PROC.poll() is not None:
                raise RuntimeError(f"Krea2 MLX sidecar exited early; see {_SIDECAR_LOG}")
            time.sleep(0.25)
        raise TimeoutError(f"Krea2 MLX sidecar did not become healthy; see {_SIDECAR_LOG}")


def _pipeline():
    global _PIPE, _PIPE_KEY
    key = (str(TRANSFORMER_PATH), "mxfp8-fused", os.environ.get("KREA2_BASE_DIR"))
    with _PIPE_LOCK:
        if _PIPE is not None and _PIPE_KEY == key:
            return _PIPE
        if not TRANSFORMER_PATH.exists():
            raise FileNotFoundError(f"Krea2 Red Mix MXFP8 transformer not found: {TRANSFORMER_PATH}")
        _PIPE = None
        _PIPE_KEY = None
        gc.collect()
        _prepare_mlx_runtime(clear_cache=True)
        from krea2.pipeline import Krea2Pipeline
        _PIPE = Krea2Pipeline(
            transformer_path=str(TRANSFORMER_PATH),
            precision="mxfp8-fused",
            base_dir=os.environ.get("KREA2_BASE_DIR"),
        )
        _PIPE_KEY = key
        return _PIPE


def _pil_to_tensor(images):
    tensors = []
    for image in images:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors, dim=0)


def _image_paths_to_tensor(paths):
    from PIL import Image

    tensors = []
    for path in paths:
        with Image.open(path) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr))
    return torch.stack(tensors, dim=0)


class Krea2MLXRedMixSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "portrait photo, natural light, detailed skin, realistic"}),
                "width": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1440, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**64 - 1, "control_after_generate": True}),
                "num_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Krea2/MLX"

    def generate(self, prompt, width, height, steps, seed, num_images):
        run_start = time.perf_counter()
        last_mark = [run_start]

        def mark(label):
            if not _timings_enabled():
                return
            now = time.perf_counter()
            print(f"[Krea2MLX] {label}: total={now - run_start:.3f}s delta={now - last_mark[0]:.3f}s", flush=True)
            last_mark[0] = now

        pbar = comfy.utils.ProgressBar(int(steps))

        def callback(step, total):
            mark(f"step {step}/{total}")
            pbar.update_absolute(step, total)

        _ensure_sidecar()
        mark("pipeline")
        response = _request_json("/generate", {
            "prompt": str(prompt).strip(),
            "width": int(width),
            "height": int(height),
            "steps": int(steps),
            "seed": int(seed),
            "num_images": int(num_images),
            "prefix": "Krea2_RedMix_sidecar",
        }, timeout=max(600, int(steps) * 120))
        mark(f"sidecar_generate timings={response.get('timings')}")
        paths = response.get("images") or []
        if not paths:
            raise RuntimeError(f"Krea2 MLX sidecar returned no images: {response}")
        tensors = _image_paths_to_tensor(paths)
        mark("pil_to_tensor")
        pbar.update_absolute(int(steps), int(steps))
        return (tensors,)


class Krea2MLXFreeCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"drop_model": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ()
    FUNCTION = "free"
    OUTPUT_NODE = True
    CATEGORY = "Krea2/MLX"

    def free(self, drop_model):
        global _PIPE, _PIPE_KEY
        if drop_model:
            with _PIPE_LOCK:
                _PIPE = None
                _PIPE_KEY = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        return {}


NODE_CLASS_MAPPINGS = {
    "Krea2MLXRedMixSampler": Krea2MLXRedMixSampler,
    "Krea2MLXFreeCache": Krea2MLXFreeCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2MLXRedMixSampler": "Krea2 Red Mix MLX Sampler",
    "Krea2MLXFreeCache": "Krea2 MLX Free Cache",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
