"""
VBench batch generation for scope-overworld (Waypoint).
Generates videos for all indoor/scenery prompts using WaypointPipeline.

Usage (from scope-overworld root):
    python src/scope_overworld/test.py [--output_dir output/vbench/videos] [--num_samples 5] [--num_frames 64] [--fps 16]
"""
import argparse
import csv
import json
import os
import random
import glob as _glob
import re

import time

import psutil
import torch
import torchvision

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.join(_SCRIPT_DIR, "..", "..")
_VBENCH_DATA  = os.path.join(_ROOT_DIR, "..", "VBench", "vbench2_beta_i2v", "vbench2_beta_i2v", "data")
_DEFAULT_JSON = os.path.join(_VBENCH_DATA, "i2v-bench-info.json")
_DEFAULT_CROP = os.path.join(_VBENCH_DATA, "crop", "1-1")
_CATEGORIES   = {"scenery", "indoor"}


def _safe(text):
    return re.sub(r'[<>:"/\\|?*]', "_", text)[:150]


def _fmt_duration(secs):
    h, m, s = int(secs // 3600), int(secs % 3600 // 60), int(secs % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _sys_stats():
    vm = psutil.virtual_memory()
    ram_used = vm.used / 1024**3
    ram_total = vm.total / 1024**3
    if torch.cuda.is_available():
        gpu_used  = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        gpu_used = gpu_total = 0.0
    return ram_used, ram_total, gpu_used, gpu_total


def generate_video(pipeline, image_path: str, caption: str, num_frames: int) -> torch.Tensor:
    """Generate a video tensor (T, H, W, 3) uint8 using the pipeline."""
    frames = []

    # First frame: reset state, condition on image and prompt
    result = pipeline(
        images=[image_path],
        prompts=[caption],
        init_cache=True,
    )
    frames.append(result["video"][0])  # (H, W, 3) float in [0, 1]

    # Remaining frames
    for _ in range(num_frames - 1):
        result = pipeline()
        frames.append(result["video"][0])

    video = torch.stack(frames)                          # (T, H, W, 3) float
    return (video * 255).clamp(0, 255).byte()            # uint8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  default=r"C:\workspace\world\Infinite-World\out\vbench\videos")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_frames",  type=int, default=64)
    parser.add_argument("--fps",         type=int, default=16)
    parser.add_argument("--vbench_json", default=_DEFAULT_JSON)
    parser.add_argument("--crop_dir",    default=_DEFAULT_CROP)
    args = parser.parse_args()

    info_json = os.path.abspath(args.vbench_json)
    crop_dir  = os.path.abspath(args.crop_dir)
    out_dir   = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path   = os.path.join(os.path.dirname(out_dir), "vbench_gen_stats.csv")
    stats_is_new = not os.path.exists(stats_path)
    stats_f = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w = csv.writer(stats_f)
    if stats_is_new:
        stats_w.writerow(["timestamp", "task_idx", "prompt", "sample_idx", "seed", "duration_s",
                          "video_count", "total_elapsed_s", "avg_s_per_video",
                          "ram_used_gb", "ram_total_gb", "gpu_used_gb", "gpu_total_gb",
                          "out_path", "status"])

    with open(info_json, encoding="utf-8") as f:
        entries = json.load(f)

    seen, prompts = set(), []
    for e in entries:
        name = e["file_name"]
        if name in seen:
            continue
        if e.get("type") not in _CATEGORIES:
            continue
        seen.add(name)
        caption = e.get("caption", os.path.splitext(name)[0])
        prompts.append((name, caption))

    total = len(prompts) * args.num_samples
    print(f"{'='*70}")
    print(f"[vbench] scope-overworld (Waypoint) VBench batch")
    print(f"[vbench] {len(prompts)} prompts x {args.num_samples} samples = {total} videos")
    print(f"[vbench] {args.num_frames} frames @ {args.fps} fps  |  categories: {sorted(_CATEGORIES)}")
    print(f"[vbench] output → {out_dir}")
    print(f"[vbench] stats  → {stats_path}")
    print(f"{'='*70}")

    # Load pipeline once
    print(f"[vbench] Loading WaypointPipeline...")
    from pipeline import WaypointPipeline
    pipeline = WaypointPipeline()
    print(f"[vbench] Pipeline ready.\n")

    done = skipped = generated = errors = 0
    ok_total_s = 0.0
    t_start = time.time()

    for task_idx, (image_name, caption) in enumerate(prompts):
        image_path = os.path.join(crop_dir, image_name)
        if not os.path.isfile(image_path):
            print(f"[vbench] SKIP: image not found — {image_path}")
            continue

        for sample_idx in range(args.num_samples):
            seed     = random.randint(0, 2**31 - 1)
            out_path = os.path.join(out_dir, f"{_safe(caption)}-{sample_idx}-{seed}.mp4")

            elapsed = time.time() - t_start
            pct     = 100 * done / total if total else 0
            eta_str = avg_str = ""
            if generated > 0:
                avg_s     = ok_total_s / generated
                remaining = (total - done) * avg_s
                eta_str   = f"  ETA {_fmt_duration(remaining)}"
                avg_str   = f"  avg {avg_s/60:.1f} min/video"
            print(f"\n{'─'*70}")
            print(f"[vbench] [{done+1}/{total}  {pct:.0f}%{eta_str}{avg_str}]  elapsed {_fmt_duration(elapsed)}")
            print(f"[vbench] prompt {task_idx+1}/{len(prompts)}  sample {sample_idx+1}/{args.num_samples}  seed {seed}")
            print(f"[vbench] {caption[:70]}")

            existing = _glob.glob(os.path.join(out_dir, f"{_safe(caption)}-{sample_idx}-*.mp4"))
            if existing:
                out_path = existing[0]
                print(f"[vbench] → SKIP (already exists)")
                skipped += 1
                done += 1
                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  "", generated, f"{elapsed:.1f}", "",
                                  "", "", "", "", out_path, "skipped"])
                stats_f.flush()
                continue

            try:
                torch.manual_seed(seed)
                st = time.time()

                video_uint8 = generate_video(pipeline, image_path, caption, args.num_frames)
                torchvision.io.write_video(out_path, video_uint8, fps=args.fps)

                duration  = time.time() - st
                ok_total_s += duration
                generated += 1

                ram_used, ram_total, gpu_used, gpu_total = _sys_stats()
                avg_s_per = ok_total_s / generated

                print(f"[vbench] ✓ saved  {os.path.basename(out_path)}")
                print(f"[vbench]   duration {_fmt_duration(duration)}  |  avg {avg_s_per/60:.1f} min/video")
                print(f"[vbench]   RAM {ram_used:.1f}/{ram_total:.0f} GB  |  GPU {gpu_used:.1f}/{gpu_total:.0f} GB")

                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  f"{duration:.1f}", generated, f"{time.time()-t_start:.1f}", f"{avg_s_per:.1f}",
                                  f"{ram_used:.2f}", f"{ram_total:.2f}", f"{gpu_used:.2f}", f"{gpu_total:.2f}",
                                  out_path, "ok"])
                stats_f.flush()

            except Exception as exc:
                print(f"[vbench] ✗ ERROR: {exc}")
                ram_used, _, gpu_used, _ = _sys_stats()
                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  "", generated, f"{time.time()-t_start:.1f}", "",
                                  f"{ram_used:.2f}", "", f"{gpu_used:.2f}", "", out_path, "error"])
                stats_f.flush()
                errors += 1

            done += 1

    elapsed_total = time.time() - t_start
    stats_f.close()
    print(f"\n{'='*70}")
    print(f"[vbench] DONE  generated={generated}  skipped={skipped}  errors={errors}")
    print(f"[vbench] total elapsed: {_fmt_duration(elapsed_total)}")
    if generated:
        print(f"[vbench] avg per video: {ok_total_s/generated/60:.1f} min")
    print(f"[vbench] videos → {out_dir}")
    print(f"[vbench] stats  → {stats_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
