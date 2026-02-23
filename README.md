# scope-overworld

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

<img width="1902" height="911" alt="Screenshot 2026-01-15 153408" src="https://github.com/user-attachments/assets/3ba71d58-e5c8-4fa7-94d0-ed3673692a82" />

[Scope](https://github.com/daydreamlive/scope) plugin providing pipelines for Overworld world models.


## Features

- Waypoint - Generate worlds using the [Waypoint-1.1-Small](https://huggingface.co/Overworld/Waypoint-1.1-Small) model

## HuggingFace

Model weights require HuggingFace authentication. See the [HuggingFace guide](https://docs.daydream.live/scope/guides/huggingface) for setup instructions.

## Install

Follow the [Scope plugins guide](https://docs.daydream.live/scope/guides/plugins) to install this plugin using the URL:

```
https://github.com/daydreamlive/scope-overworld.git
```

## Upgrade

Follow the [Scope plugins guide](https://docs.daydream.live/scope/guides/plugins) to upgrade this plugin to the latest version.

## Architecture

The `waypoint` pipeline uses [world_engine](https://github.com/Wayfarer-Labs/world_engine) for inference. It loads three model components: the Waypoint world model, OWL VAE (encoder/decoder), and UMT5-XL (text encoder). On first load, a JIT warmup pass runs for optimized performance which could take as long as 20 minutes on the first run.

Each frame: controller input and prompt are processed → world_engine generates the next frame → the pipeline outputs a video tensor.

