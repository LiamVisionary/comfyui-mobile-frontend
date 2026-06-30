# Krea2 Apple Silicon Optimization Findings

Last updated: 2026-06-30

This document tracks Apple Silicon optimization work for the native Krea2 Red Mix MLX workflow so we do not repeat dead-end experiments. Mark claims as confirmed or inferred.

## Current Baseline

Confirmed from the native sidecar profile at 960x1440, 10 steps, no LoRA:

- Workflow: `/Users/liam/comfy/ComfyUI/user/default/workflows/Krea2 Red Mix SeedVR2 Apple Silicon.json`
- Native sampler node: `Krea2MLXRedMixSampler`
- Native sidecar: `/Users/liam/comfy/ComfyUI/custom_nodes/krea2_mlx_redmix/sidecar.py`
- MLX pipeline: `/Users/liam/comfy/krea2_alis_mlx_redmix/krea2/pipeline.py`
- Sampler implementation: `/Users/liam/comfy/krea2_alis_mlx_redmix/krea2/sampling.py`
- Transformer precision: `mxfp8-fused`
- Active fast path: fused attention, fused MLP, compiled `forward_prepared_vectors`
- Normal activation dtype: `bf16`
- Generate time: about 50-55s cold for 960x1440 / 10 steps / no LoRA under current conditions
- Stage profile: denoise is about 48.4s of about 50.6s generate; text encode is about 0.14s; VAE decode is about 1.9s

Conclusion: the bottleneck is the 28-block transformer denoise pass over the 960x1440 token grid. Text encoding, VAE decode, image save, and workflow plumbing are not the main problem.

Lightweight benchmark target added on 2026-06-30:

- Resolution: 256x384, the same 2:3 aspect ratio as 960x1440.
- Steps: 10.
- Seed: 794015397137290.
- Sampler: `flow_euler`.
- LoRA: disabled unless explicitly testing LoRA.
- Baseline long saved JSON prompt: about 6.1-6.8s warm sidecar generate under clean-ish conditions, but can drift under system load.
- The reference PNG `/Users/liam/Downloads/e39e3b884e724eb8bb19e6176a408f42.png` embeds the same base seed `794015397137290`, a separate SeedVR2 upscaler seed `3041761909`, and final size 1920x2880.
- Same seed across different resolutions is not expected to produce the same image. Krea2 samples a different latent tensor shape, different image token count, different RoPE positions, and a different timestep schedule when resolution changes.

## What Sampler Are We Using?

Confirmed from `/Users/liam/comfy/krea2_alis_mlx_redmix/krea2/sampling.py`:

- The native Krea2 MLX path does not use Comfy's `KSampler`, `sampler_name`, or `scheduler` controls.
- It uses a bespoke Krea-2 flow-matching sampler ported from `krea-2-official/sampling.py`.
- The actual update is Euler-style:

```python
img = img + (tp - tc) * v
```

- Timesteps come from `timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None)`.
- The native sidecar now exposes `sampler` with `flow_euler` and experimental `er_sde`; the saved workflow remains on `flow_euler` because the first `er_sde` benchmark was not faster or visually preferable.
- The pre-native workflow `/Users/liam/comfy/ComfyUI/user/default/workflows/Krea2 Red Mix SeedVR2 Apple Silicon.pre-mlx-native-20260629-095508.json` did use Comfy `KSampler` with widgets:

```json
[794015397137290, "randomize", 10, 1, "er_sde", "simple", 1]
```

- Comfy registers `er_sde` in `/Users/liam/comfy/ComfyUI/comfy/samplers.py`.
- Comfy's implementation is `/Users/liam/comfy/ComfyUI/comfy/k_diffusion/sampling.py::sample_er_sde`.
- Comfy's custom sampler node wraps it in `/Users/liam/comfy/ComfyUI/comfy_extras/nodes_custom_sampler.py::SamplerER_SDE`.

Recommendation:

- For workflows that still use Comfy `KSampler`, prefer `er_sde` with a simple scheduler for speed/quality tradeoff testing.
- For the native Krea2 MLX sidecar, `er_sde` is selectable as an experiment, but it is not the recommended default based on the first benchmark.
- Do not claim the native Krea2 workflow is "using er_sde" unless the saved workflow's `Krea2MLXRedMixSampler` widget value is explicitly set to `er_sde`.
- ER-SDE is attractive because Comfy's implementation uses one model call per step. That means it is sampler-efficient compared with multi-eval samplers.
- Porting ER-SDE is not copy/paste: Comfy's version assumes sigma/logSNR data-prediction sampling, while native Krea2 uses flow-matching velocity prediction with direct `t` timesteps. We need a Krea2-correct derivation or an empirical adapter before enabling it.
- First native adapter result: runtime-stable, about 54.2s generate at 960x1440 / 10 steps / no LoRA, not a speed win.

## Working Changes To Keep

Confirmed improvements or useful instrumentation:

- Keep `mxfp8-fused` transformer weights.
- Keep fused attention and fused MLP projections.
- Keep `mx.compile(transformer.forward_prepared_vectors)`.
- Keep `MLX_METAL_FAST_SYNCH=1`.
- Keep `KREA2_MLX_CACHE_LIMIT_GB=0`; this disables MLX's free-memory cache. It does not disable compiled forward kernels.
- Keep `KREA2_MLX_EVAL_EACH_STEP=0` and `KREA2_MLX_STEP_TIMINGS=0` for normal generation. Step timings force synchronization and are for profiling only.
- Keep sidecar health reporting for:
  - `precision`
  - fused attention / fused MLP
  - compiled forward wrapper
  - activation dtype
  - MLX cache limit
  - active LoRA count
- Keep stale sidecar rejection in the Comfy node so old listeners are not silently reused.
- Keep `KREA2_MLX_RECYCLE_SIDECAR_AFTER_RUN=1` for now. Warm repeats in one loaded process drifted from about 50s to about 90-130s.
- Keep `start_new_session=True` when spawning the sidecar so it survives the launcher process correctly.
- Do not demote the sidecar with `taskpolicy -B` unless explicitly setting `KREA2_MLX_BACKGROUND_SIDECAR=1`.

## Dead Ends / Do Not Retry Blindly

Confirmed non-working or worse:

- Reducing steps from 10 to 4 is not acceptable; quality visibly degrades.
- Native experimental `er_sde` at 960x1440 / 10 steps / no LoRA was not faster. It took about 54.2s generate versus the about 50-55s current `flow_euler` range, and the visual output changed composition enough that it should not be the default.
- Setting MLX free-memory cache to 12 GB made warm repeats worse in testing, including a run around 91s.
- Activation dtype `fp16` was slower than `bf16` in testing, around 90s generate.
- Generic FP8 activations are not a drop-in MLX dtype. MLX exposes `to_fp8` and `from_fp8` conversion helpers that store FP8 as `uint8`; using them for intermediate activations would add quantize/dequantize overhead and likely hurt quality.
- Directly calling `linear(x)` on 3D tensors instead of the existing flatten-and-reshape helper was slower, around 60.9s generate, and was reverted.
- Capturing `transformer.parameters()` explicitly in `mx.compile(..., inputs=...)` failed with `ValueError: [compile] Attempting to compile a function with uncaptured inputs is not allowed.`
- The text encoder and VAE are not the main bottleneck for this workflow at the tested settings.
- MLX `fast.rms_norm` with float32 inputs/effective weight preserved did not improve the target run. It took about 52.2s generate / 59.0s wall at 960x1440 / 10 steps / no LoRA and was reverted.
- MLX `fast.rms_norm` in native bf16 was much worse, about 99.9s generate / 105.5s wall at 960x1440 / 10 steps / no LoRA. Do not retry without a specific MLX kernel/runtime change.
- Red Mix bf16 fused-after-load was slower than the MXFP8 fused path, about 104.3s generate at 960x1440 / 10 steps / no LoRA. Higher precision is not a speed win on this machine.
- Red Mix MXFP8 unfused was slower than the fused MXFP8 path, about 113.0s generate at 960x1440 / 10 steps / no LoRA. The fused QKV/gate and MLP gate/up packing should stay.
- MLX cache limit 2 GB gave only a small/noisy improvement, about 49.4s generate. MLX cache limit 4 GB was much worse, about 95.9s generate. Keep normal default at 0 GB unless repeated evidence says 2 GB is stable.
- Compiling the entire fixed denoise loop, instead of only `forward_prepared_vectors`, was worse at target shape: about 58.9s denoise on the first compiled call and about 74.5s on the second.
- Forcing `mx.eval` after every denoise step was worse, about 83.2s generate. The lazy 10-step graph should remain.
- At 256x384, compiling the whole 10-step denoise loop was also worse than the current per-step compiled-forward sidecar path. Do not wire whole-loop compile for small shapes.
- Prompt compaction is a real speed lever at 256x384 because the original saved prompt fills/truncates to 512 text tokens while compact prompts reduce the transformer context length. This is a conditioning change, not a pure runtime/kernel optimization. Use only if visually accepted.
- Accepted compact prompt candidate, 537 chars, produced about 3.6-5.0s warm sidecar generate at 256x384 depending on system load:
  `photorealistic close-up overhead selfie of an adult young East Asian woman in her early 20s with fair skin, short black bob haircut and blunt bangs, lying in bright clean sunlight, playful cheeky expression, one eye open looking at the camera and the other eye winking, soft smile with glossy coral lips, white camisole straps, wicker chair texture, arms and hands very close to the lens creating foreground occlusion, intimate POV framing, diagonal window-blind shadows across face and body, natural skin texture, sharp realistic detail`
- Very short/tuned prompts did not reliably improve beyond the accepted compact prompt and often changed output or slowed down; do not assume fewer characters is faster.
- Dynamic text length is default-off. It can change Qwen text-encoder positions for compact prompts, so it is an experiment, not a preservation-safe default.
- `MLX_ENABLE_TF32=1` is slower on the 256x384 fixed-seed long-prompt benchmark. Warm generate was about 7.24-8.50s versus the 6.35-6.38s baseline.
- `MLX_METAL_JIT=0` is much slower on the 256x384 fixed-seed long-prompt benchmark. Warm generate drifted from about 10.99s to 19.33s.
- `MLX_METAL_JIT=1` is much slower on the 256x384 fixed-seed long-prompt benchmark. Warm generate was about 16.18-17.71s after a 31.12s first generate.
- `MLX_ENABLE_TF32=1` plus `MLX_METAL_JIT=1` is slower on the 256x384 fixed-seed long-prompt benchmark. Warm generate was about 14.14-14.44s.
- Do not retry TF32 or explicit `MLX_METAL_JIT` as a generic speed fix unless MLX changes or the workload changes. The sidecar fast-path report now records both flags so stale-runtime comparisons are visible.
- Progressive latent sampling is a real speed lever but failed the preserved-quality constraint in the first full-size tests. A 960x1440 final image with 6 early steps at 608x896 and 4 final steps at 960x1440 reached about 30.05s denoise / 32.84s after load, but produced visible color artifacts and wardrobe/content drift. Safer 608x896/4 and 720x1088/6 variants were slower, about 51-55s denoise in the tested run, and still changed the image. Do not wire progressive latent into the main workflow unless a later variant removes those artifacts.

## Open Hypotheses Worth Testing

Potential next experiments that preserve 960x1440, 10 steps, and model quality:

- Tune the native `er_sde` experiment only if there is a specific quality target. A first deterministic ER-SDE-style adapter exists, but it is not faster at equal steps.
- Try fewer-step `er_sde` only as a separate quality experiment. It may be useful if it preserves quality at fewer steps, but equal-step speed is not better.
- Add richer sampler controls only after a sampler beats or clearly complements `flow_euler`; avoid cluttering the workflow with unproven knobs.
- Investigate whether MLX has a lower-overhead compiled loop for the full denoise sequence, not only `forward_prepared_vectors` per step.
- Compile or specialize the whole denoise function for fixed shape `(960, 1440, 10)` if MLX can capture the transformer module safely without uncaptured-input errors.
- Profile individual transformer block costs with coarse synchronization every N blocks to see whether attention, MLP, RMSNorm, or modulation dominates.
- Profile one target-resolution denoise step with intra-block synchronization to separate fused qkvgate, q/k norm, RoPE, scaled dot-product attention, output projection, and MLP cost. This is diagnostic only; do not compare its absolute time to normal unsynchronized runs.
- Confirmed diagnostic profile at 960x1440 target shape, one warmed forward, uncompiled and synchronized inside blocks: MLP total was about 5.4s, attention total about 4.7s, qkvgate about 1.5s, SDPA about 1.1s, RoPE about 0.8s. Broadly, the cost is the main transformer matmuls/attention over 5912 tokens, not text encode, VAE, or workflow plumbing.
- Test whether shape-specific resolution buckets with fewer image tokens but equivalent final upscaling can preserve quality better than lowering steps. This is not a first choice because the user does not want quality degradation.
- Explore MLX Metal captures for the denoise pass to identify kernel stalls or unexpected CPU sync points.
- Investigate LoRA adapter overhead separately. Existing logs show one LoRA can push generate time into about 80-116s and two LoRAs to about 130s.
- For standard LoRAs, investigate adapter fusion or pre-packed multi-adapter projections without dequantizing/replacing the MXFP8 base weights.
- If revisiting progressive latent, test it as an explicit opt-in turbo mode only. The first viable speed setting hit near-target runtime but failed visual QA; the next attempt should focus on artifact-free latent resize/timestep handoff, not just lower token counts.

## Benchmark Rules

Use the same comparison setup unless explicitly testing another axis:

- Resolution: 960x1440
- Steps: 10
- LoRA: disabled for baseline unless testing LoRA overhead
- Seed: `794015397137290`
- Prompt: saved prompt in `Krea2 Red Mix SeedVR2 Apple Silicon.json`
- Report:
  - wall time
  - sidecar `timings.generate`
  - profile stages if `KREA2_MLX_PROFILE_STAGES=1`
  - sidecar fast-path report
  - whether run was cold or warm

## Useful Environment Flags

- `KREA2_MLX_PROFILE_STAGES=1`: opt-in stage profiling; slower because it synchronizes stage boundaries.
- `KREA2_MLX_STEP_TIMINGS=1`: opt-in per-step timings; slower because it synchronizes every step.
- `KREA2_MLX_EVAL_EACH_STEP=1`: old forced sync behavior; use only for debugging.
- `KREA2_MLX_ACTIVATION_DTYPE=bf16`: default and fastest confirmed safe activation dtype.
- `KREA2_MLX_ACTIVATION_DTYPE=fp16`: available for testing; slower in current benchmark.
- `KREA2_MLX_RECYCLE_SIDECAR_AFTER_RUN=1`: default to avoid warm-run drift.
- `KREA2_MLX_BACKGROUND_SIDECAR=1`: opt into background scheduling; default is foreground because background scheduling was a suspected slowdown.
