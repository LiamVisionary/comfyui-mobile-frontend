import gc
import os
import sys
import threading
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
        pbar = comfy.utils.ProgressBar(int(steps))

        def callback(step, total):
            pbar.update_absolute(step, total)

        pipe = _pipeline()
        _prepare_mlx_runtime(clear_cache=True)
        images = pipe.generate(
            str(prompt).strip(),
            width=int(width),
            height=int(height),
            steps=int(steps),
            seed=int(seed),
            num_images=int(num_images),
            step_callback=callback,
        )
        return (_pil_to_tensor(images),)


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
