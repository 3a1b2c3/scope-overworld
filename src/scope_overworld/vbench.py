"""
Batch video generation for scope-overworld (Waypoint).
Generates N videos per prompt, saves to output_dir, writes per-video CSV stats.

Usage (from scope-overworld root):
    # With VBench dataset:
    python src/scope_overworld/vbench.py --vbench_json ../VBench/.../i2v-bench-info.json --crop_dir ../VBench/.../crop/1-1

    # With a plain prompts file (one prompt per line):
    python src/scope_overworld/vbench.py --prompts_file prompts.txt [--images_dir images/]

    # Minimal — generates from built-in default prompts, no images:
    python src/scope_overworld/vbench.py
"""
import os
import sys

# Re-launch with .venv Python if not already running inside it
_venv_python = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".venv", "Scripts", "python.exe"
)
if os.path.exists(_venv_python) and os.path.abspath(sys.executable) != os.path.abspath(_venv_python):
    import subprocess
    sys.exit(subprocess.run([_venv_python] + sys.argv).returncode)

import argparse
import csv
import json
import random
import re
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import psutil
import torch
import torchvision

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.join(_SCRIPT_DIR, "..", "..")

_DEFAULT_PROMPTS = [
    "a mountain landscape with snow",
    "a city street at night with lights",
    "a sandy beach with waves",
    "a dense green forest",
    "a cozy indoor living room",
]
_VBENCH_CATEGORIES = {"scenery", "indoor"}


def _safe(text):
    return re.sub(r'[<>:"/\\|?*]', "_", text)[:150]


def _fmt_duration(secs):
    h, m, s = int(secs // 3600), int(secs % 3600 // 60), int(secs % 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def _sys_stats():
    vm = psutil.virtual_memory()
    ram_used  = vm.used  / 1024**3
    ram_total = vm.total / 1024**3
    if torch.cuda.is_available():
        gpu_used  = torch.cuda.memory_allocated() / 1024**3
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        gpu_used = gpu_total = 0.0
    return ram_used, ram_total, gpu_used, gpu_total


def load_prompts(args):
    """Returns list of (image_path_or_None, caption) tuples."""
    # Priority 1: VBench JSON
    if args.vbench_json and os.path.isfile(args.vbench_json):
        crop_dir = os.path.abspath(args.crop_dir) if args.crop_dir else None
        with open(args.vbench_json, encoding="utf-8") as f:
            entries = json.load(f)
        seen, result = set(), []
        for e in entries:
            name = e["file_name"]
            if name in seen:
                continue
            if e.get("type") not in _VBENCH_CATEGORIES:
                continue
            seen.add(name)
            caption = e.get("caption", os.path.splitext(name)[0])
            image_path = os.path.join(crop_dir, name) if crop_dir else None
            result.append((image_path, caption))
        return result

    # Priority 2: plain text prompts file
    if args.prompts_file and os.path.isfile(args.prompts_file):
        captions = [l.strip() for l in open(args.prompts_file, encoding="utf-8") if l.strip()]
        images_dir = os.path.abspath(args.images_dir) if args.images_dir else None
        result = []
        for caption in captions:
            image_path = None
            if images_dir:
                stem = _safe(caption)
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    candidate = os.path.join(images_dir, stem + ext)
                    if os.path.isfile(candidate):
                        image_path = candidate
                        break
            result.append((image_path, caption))
        return result

    # Priority 3: built-in defaults
    return [(None, p) for p in _DEFAULT_PROMPTS]


def generate_video(pipeline, image_path, caption: str, num_frames: int) -> torch.Tensor:
    """Generate a video tensor (T, H, W, 3) uint8."""
    frames = []
    call_kwargs = {"prompts": [caption], "init_cache": True}
    if image_path and os.path.isfile(image_path):
        call_kwargs["images"] = [image_path]

    result = pipeline(**call_kwargs)
    frames.append(result["video"][0])  # (H, W, 3) float [0, 1]

    for _ in range(num_frames - 1):
        result = pipeline()
        frames.append(result["video"][0])

    video = torch.stack(frames)
    return (video * 255).clamp(0, 255).byte()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",   default=os.path.join(_ROOT_DIR, "out", "videos"))
    parser.add_argument("--num_samples",  type=int, default=5)
    parser.add_argument("--num_frames",   type=int, default=161)
    parser.add_argument("--fps",          type=int, default=16)
    # VBench dataset (optional)
    parser.add_argument("--vbench_json",  default=None, help="Path to i2v-bench-info.json")
    parser.add_argument("--crop_dir",     default=None, help="VBench crop image directory")
    # Plain file mode (optional)
    parser.add_argument("--prompts_file", default=None, help="Text file with one prompt per line")
    parser.add_argument("--images_dir",   default=None, help="Directory of conditioning images (matched by prompt stem)")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path   = os.path.join(os.path.dirname(out_dir), "gen_stats.csv")
    stats_is_new = not os.path.exists(stats_path)
    stats_f = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w = csv.writer(stats_f)
    if stats_is_new:
        stats_w.writerow(["timestamp", "task_idx", "prompt", "sample_idx", "seed",
                          "duration_s", "gen_fps", "video_count", "total_elapsed_s", "avg_s_per_video",
                          "ram_used_gb", "ram_total_gb", "gpu_used_gb", "gpu_total_gb",
                          "out_path", "status"])
        stats_f.flush()

    prompts = load_prompts(args)
    total   = len(prompts) * args.num_samples

    print(f"{'='*70}")
    print(f"[gen] scope-overworld (Waypoint) batch generation")
    print(f"[gen] {len(prompts)} prompts x {args.num_samples} samples = {total} videos")
    print(f"[gen] {args.num_frames} frames @ {args.fps} fps")
    print(f"[gen] output → {out_dir}")
    print(f"[gen] stats  → {stats_path}")
    print(f"{'='*70}")

    print(f"[gen] Loading WaypointPipeline...")
    from scope_overworld.pipeline import WaypointPipeline
    pipeline = WaypointPipeline()
    print(f"[gen] Pipeline ready.\n")

    done = skipped = generated = errors = 0
    ok_total_s = 0.0
    t_start = time.time()

    for task_idx, (image_path, caption) in enumerate(prompts):
        for sample_idx in range(args.num_samples):
            seed     = random.randint(0, 2**31 - 1)
            out_path = os.path.join(out_dir, f"{_safe(caption)}-{sample_idx}.mp4")

            elapsed = time.time() - t_start
            pct     = 100 * done / total if total else 0
            eta_str = avg_str = ""
            if generated > 0:
                avg_s     = ok_total_s / generated
                remaining = (total - done) * avg_s
                eta_str   = f"  ETA {_fmt_duration(remaining)}"
                avg_str   = f"  avg {avg_s/60:.1f} min/video"
            print(f"\n{'─'*70}")
            print(f"[gen] [{done+1}/{total}  {pct:.0f}%{eta_str}{avg_str}]  elapsed {_fmt_duration(elapsed)}")
            print(f"[gen] prompt {task_idx+1}/{len(prompts)}  sample {sample_idx+1}/{args.num_samples}  seed {seed}")
            print(f"[gen] {caption[:70]}")

            if os.path.isfile(out_path):
                print(f"[gen] → SKIP (already exists)")
                skipped += 1
                done += 1
                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  "", "", generated, f"{elapsed:.1f}", "",
                                  "", "", "", "", out_path, "skipped"])
                stats_f.flush()
                continue

            try:
                torch.manual_seed(seed)
                st = time.time()

                video_uint8 = generate_video(pipeline, image_path, caption, args.num_frames)
                torchvision.io.write_video(out_path, video_uint8, fps=args.fps)

                duration   = time.time() - st
                ok_total_s += duration
                generated  += 1
                gen_fps     = args.num_frames / duration if duration > 0 else 0.0

                ram_used, ram_total, gpu_used, gpu_total = _sys_stats()
                avg_s_per = ok_total_s / generated

                print(f"[gen] saved  {os.path.basename(out_path)}")
                print(f"[gen]   duration {_fmt_duration(duration)}  |  {gen_fps:.2f} gen-fps  |  avg {avg_s_per/60:.1f} min/video")
                print(f"[gen]   RAM {ram_used:.1f}/{ram_total:.0f} GB  |  GPU {gpu_used:.1f}/{gpu_total:.0f} GB")

                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  f"{duration:.1f}", f"{gen_fps:.2f}", generated, f"{time.time()-t_start:.1f}", f"{avg_s_per:.1f}",
                                  f"{ram_used:.2f}", f"{ram_total:.2f}", f"{gpu_used:.2f}", f"{gpu_total:.2f}",
                                  out_path, "ok"])
                stats_f.flush()

            except Exception as exc:
                print(f"[gen] ERROR: {exc}")
                ram_used, _, gpu_used, _ = _sys_stats()
                stats_w.writerow([time.strftime("%Y-%m-%dT%H:%M:%S"), task_idx, caption, sample_idx, seed,
                                  "", "", generated, f"{time.time()-t_start:.1f}", "",
                                  f"{ram_used:.2f}", "", f"{gpu_used:.2f}", "", out_path, "error"])
                stats_f.flush()
                errors += 1

            done += 1

    elapsed_total = time.time() - t_start
    stats_f.close()
    print(f"\n{'='*70}")
    print(f"[gen] DONE  generated={generated}  skipped={skipped}  errors={errors}")
    print(f"[gen] total elapsed: {_fmt_duration(elapsed_total)}")
    if generated:
        print(f"[gen] avg per video: {ok_total_s/generated/60:.1f} min")
    print(f"[gen] videos → {out_dir}")
    print(f"[gen] stats  → {stats_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
