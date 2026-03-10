"""
Standalone test script for WaypointPipeline.
Run from the scope-overworld root:
    python test_pipeline.py [--prompt "..."] [--frames 10] [--image path.png] [--save out.mp4]
"""
import argparse
import os
import sys

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",  default="A fun game", help="Text prompt")
    p.add_argument("--frames",  type=int, default=10,  help="Frames to generate")
    p.add_argument("--image",   default=None,           help="Optional conditioning image path")
    p.add_argument("--warmup",  type=int, default=3,    help="Warmup frames")
    p.add_argument("--save",    default=None,           help="Save output to .mp4 (requires imageio[ffmpeg])")
    p.add_argument("--fps",     type=int, default=8,    help="FPS when saving video")
    p.add_argument("--device",  default=None,           help="cuda / cpu (auto-detect if unset)")
    return p.parse_args()


def main():
    args = parse_args()

    # Add src/ to path so scope_overworld is importable without install
    src_dir = os.path.join(os.path.dirname(__file__), "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    from scope_overworld.pipeline import WaypointPipeline
    from scope.core.pipelines.controller import CtrlInput

    device = torch.device(args.device) if args.device else None

    print(f"[test] Loading WaypointPipeline  prompt={args.prompt!r}  warmup={args.warmup}")
    pipeline = WaypointPipeline(
        prompt=args.prompt,
        device=device,
        warmup_frames=args.warmup,
    )
    print("[test] Pipeline ready")

    frames = []
    for i in range(args.frames):
        call_kwargs = {"ctrl_input": CtrlInput()}

        if args.image and i == 0:
            call_kwargs["images"] = [args.image]
            call_kwargs["init_cache"] = True

        out = pipeline(**call_kwargs)
        # out["video"]: (1, H, W, 3) float32 [0, 1]
        frame = out["video"][0]  # (H, W, 3)
        frames.append(frame)

        hw = f"{frame.shape[0]}x{frame.shape[1]}"
        print(f"[test] frame {i+1}/{args.frames}  shape={hw}  min={frame.min():.3f}  max={frame.max():.3f}")

    if args.save:
        try:
            import imageio
        except ImportError:
            print("[test] imageio not installed — run: pip install imageio[ffmpeg]")
            sys.exit(1)

        # Convert to uint8 numpy (T, H, W, 3)
        video_np = np.stack(
            [(f.cpu().numpy() * 255).clip(0, 255).astype(np.uint8) for f in frames]
        )
        imageio.mimwrite(args.save, video_np, fps=args.fps, codec="libx264", quality=8)
        print(f"[test] Saved {len(frames)} frames → {args.save}")
    else:
        print(f"[test] Done. {len(frames)} frames generated (use --save out.mp4 to save)")


if __name__ == "__main__":
    main()
