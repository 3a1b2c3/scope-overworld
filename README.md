# scope-overworld

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

<img width="1902" height="911" alt="Screenshot 2026-01-15 153408" src="https://github.com/user-attachments/assets/3ba71d58-e5c8-4fa7-94d0-ed3673692a82" />

[Scope](https://github.com/daydreamlive/scope) plugin providing pipelines for Overworld world models.


## Features

- **Waypoint 1.5** — Generate worlds at 720p / up to 60 fps using the [Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B) model (Apache-2.0).
- **Waypoint 1.5 (360p)** — Lighter-weight variant for laptop GPUs and Apple Silicon via [Waypoint-1.5-1B-360P](https://huggingface.co/Overworld/Waypoint-1.5-1B-360P).

## Hardware guidance

| Variant | Target hardware | Approx. FPS |
|---|---|---|
| Waypoint 1.5 (720p) | RTX 5090 | 56 fps unquantized, 72 fps with `fp8w8a8` |
| Waypoint 1.5 (720p) | RTX 3090 | ~30 fps with `intw8a8` |
| Waypoint 1.5 (360p) | Laptop GPUs, Apple Silicon | Real-time up to 60 fps |

Quantization options exposed in the UI (load-time setting):
- `intw8a8` — INT8 weights/activations, requires NVIDIA Ampere+ (30xx)
- `fp8w8a8` — FP8, requires Ada Lovelace / Hopper+
- `nvfp4` — NVFP4, requires Blackwell

## HuggingFace

Model weights may require HuggingFace authentication. See the [HuggingFace guide](https://docs.daydream.live/scope/guides/huggingface) for setup instructions.

## Install

Follow the [Scope plugins guide](https://docs.daydream.live/scope/guides/plugins) to install this plugin using the URL:

```
https://github.com/daydreamlive/scope-overworld.git
```

## Upgrade

Follow the [Scope plugins guide](https://docs.daydream.live/scope/guides/plugins) to upgrade this plugin to the latest version.

## Architecture

Both `waypoint` and `waypoint_360p` pipelines use [world_engine](https://github.com/Overworldai/world_engine) for inference. The model is an autoregressive Diffusion Transformer with a bundled Tiny Hunyuan Autoencoder (`taehv1_5`) providing 4× temporal and 8× spatial compression. On first load, a JIT warmup pass runs to trigger compilation.

Each inference step: controller input (keyboard/mouse) and the active prompt are processed → `world_engine` generates the next 4-frame chunk at the target resolution → Scope's pipeline processor splits the chunk into per-frame packets for the output stream.

Context window: 512 frames (~10 seconds at 60 fps).
