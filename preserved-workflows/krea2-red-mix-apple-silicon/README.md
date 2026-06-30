# Krea2 Red Mix Apple Silicon Baseline

This folder preserves the working local Krea2 Red Mix workflow and native MLX fast path as of 2026-06-29.

Runtime source paths:

- Workflow: `/Users/liam/comfy/ComfyUI/user/default/workflows/Krea2 Red Mix SeedVR2 Apple Silicon.json`
- Native Comfy node: `/Users/liam/comfy/ComfyUI/custom_nodes/krea2_mlx_redmix/__init__.py`
- Native MLX sidecar: `/Users/liam/comfy/ComfyUI/custom_nodes/krea2_mlx_redmix/sidecar.py`
- Krea2 MLX checkout: `/Users/liam/comfy/krea2_alis_mlx_redmix`
- ComfyUI history privacy patch: `/Users/liam/comfy/ComfyUI/execution.py`
- LaunchAgent scheduling config: `/Users/liam/Library/LaunchAgents/com.liam.zimage-stack.plist`

The workflow is saved with the current quality-preserving settings: 960x1440, 10 steps, seed `794015397137290`, and the reference-style prompt used for the latest comparison run.

The Comfy node delegates Krea2 generation to a local pure-MLX sidecar on port `8796`.
The sidecar defaults `MLX_METAL_FAST_SYNCH=1`, `KREA2_MLX_COMPILE_FORWARD=1`, and `KREA2_MLX_CACHE_LIMIT_GB=0` to keep the hot transformer pass on the faster Apple Silicon path without changing prompt, model, resolution, or step count.
The LaunchAgent is saved with `ProcessType=Standard`; `Interactive` was rejected by launchd on this machine, while `Background` risked throttling the user-facing generation stack.
On the saved 960x1440, 10-step, full-prompt workflow, the best current sidecar profile reports step timings around 4.0-5.2 seconds and about 47.8 seconds total generation time when warm.
Through the real Comfy API path after the `Standard` reload, one cold sidecar run completed in 53.4 seconds wall time; a subsequent repeat under different system load took 67.5 seconds.

Only source files, workflow JSON, and a small ComfyUI patch are preserved here. Model weights and generated outputs are intentionally excluded.
