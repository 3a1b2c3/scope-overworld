"""
Batch video generation for scope-overworld (Waypoint).
Generates N videos per prompt, saves to output_dir, writes per-video CSV stats.

Usage (from scope-overworld root):
    # With VBench dataset:
    python src/scope_overworld/vbench.py --vbench_info_json ../VBench/.../i2v-bench-info.json --crop_dir ../VBench/.../crop/1-1

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

import csv
import json
import re
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import warnings
warnings.filterwarnings("always")
import torch
import torch._dynamo

import argparse
import psutil
import torch
import torchvision

_SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR          = os.path.join(_SCRIPT_DIR, "..", "..")
_VBENCH_DATA       = os.path.join(_ROOT_DIR, "..", "VBench", "vbench2_beta_i2v", "vbench2_beta_i2v", "data")
_DEFAULT_INFO_JSON = os.path.join(_VBENCH_DATA, "i2v-bench-info.json")
_DEFAULT_CROP_DIR  = os.path.join(_VBENCH_DATA, "crop", "1-1")

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


def load_prompts(vbench_info_json, crop_dir, prompts_file, images_dir):
    """Returns list of (image_path_or_None, caption) tuples."""
    # Priority 1: VBench JSON
    if vbench_info_json and os.path.isfile(vbench_info_json):
        crop_base = os.path.abspath(crop_dir) if crop_dir else None
        with open(vbench_info_json, encoding="utf-8") as f:
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
            image_path = os.path.join(crop_base, name) if crop_base else None
            result.append((image_path, caption))
        return result

    # Priority 2: plain text prompts file
    if prompts_file and os.path.isfile(prompts_file):
        captions = [l.strip() for l in open(prompts_file, encoding="utf-8") if l.strip()]
        images_base = os.path.abspath(images_dir) if images_dir else None
        result = []
        for caption in captions:
            image_path = None
            if images_base:
                stem = _safe(caption)
                for ext in (".jpg", ".jpeg", ".png", ".webp"):
                    candidate = os.path.join(images_base, stem + ext)
                    if os.path.isfile(candidate):
                        image_path = candidate
                        break
            result.append((image_path, caption))
        return result

    # Priority 3: built-in defaults
    return [(None, p) for p in _DEFAULT_PROMPTS]


def generate_video(pipeline, image_path, caption: str, num_frames: int, ctrl_input=None) -> torch.Tensor:
    """Generate a video tensor (T, H, W, 3) uint8."""
    from scope.core.pipelines.controller import CtrlInput
    ctrl = ctrl_input or CtrlInput()

    frames = []
    call_kwargs = {"prompts": [caption], "init_cache": True, "ctrl_input": ctrl}
    if image_path and os.path.isfile(image_path):
        call_kwargs["images"] = [image_path]

    result = pipeline(**call_kwargs)
    frames.append(result["video"][0])  # (H, W, 3) float [0, 1]

    for _ in range(num_frames - 1):
        result = pipeline(ctrl_input=ctrl)
        frames.append(result["video"][0])

    video = torch.stack(frames)
    return (video * 255).clamp(0, 255).byte()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",       default=os.path.join(_ROOT_DIR, "out", "videos"))
    parser.add_argument("--num_samples",      type=int, default=5)
    parser.add_argument("--num_frames",       type=int, default=161)
    parser.add_argument("--fps",              type=int, default=24)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--vbench_info_json", default=_DEFAULT_INFO_JSON, help="Path to i2v-bench-info.json")
    parser.add_argument("--crop_dir",         default=_DEFAULT_CROP_DIR, help="VBench crop image directory")
    parser.add_argument("--prompts_file",     default=None, help="Text file with one prompt per line")
    parser.add_argument("--images_dir",       default=None, help="Directory of conditioning images")
    parser.add_argument("--ctrl",             default=None, help="Comma-separated W3C key codes to hold (e.g. KeyW,Space)")
    args = parser.parse_args()

    output_dir       = args.output_dir
    num_samples      = args.num_samples
    num_frames       = args.num_frames
    fps              = args.fps
    seed             = args.seed
    vbench_info_json = args.vbench_info_json
    crop_dir         = args.crop_dir
    prompts_file     = args.prompts_file
    images_dir       = args.images_dir

    from scope.core.pipelines.controller import CtrlInput
    if args.ctrl:
        keys = {k.strip() for k in args.ctrl.split(",") if k.strip()}
        ctrl_input = CtrlInput(button=keys)
    else:
        ctrl_input = CtrlInput()

    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    stats_path    = os.path.join(os.path.dirname(out_dir), "vbench_stats.csv")
    _stats_is_new = not os.path.exists(stats_path)
    stats_f       = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w       = csv.writer(stats_f)
    if _stats_is_new:
        stats_w.writerow(["task_idx", "prompt", "sample_idx", "duration_s", "gen_fps",
                          "ram_gb", "vram_gb", "out_path", "status"])
        stats_f.flush()

    prompts = load_prompts(vbench_info_json, crop_dir, prompts_file, images_dir)
    total   = len(prompts) * num_samples

    print(f"{'='*70}")
    print(f"[vbench] scope-overworld (Waypoint) batch generation")
    print(f"[vbench] {len(prompts)} prompts × {num_samples} samples = {total} videos")
    print(f"[vbench] {num_frames} frames @ {fps} fps")
    print(f"[vbench] output → {out_dir}")
    print(f"[vbench] stats  → {stats_path}")
    print(f"{'='*70}")

    print(f"[vbench] Loading WaypointPipeline...")
    from scope_overworld.pipeline import WaypointPipeline
    pipeline = WaypointPipeline()
    print(f"[vbench] Pipeline ready.\n")

    skipped = generated = errors = 0
    done = 0
    ok_total_s = 0.0
    t_start = time.time()

    for task_idx, (image_path, prompt) in enumerate(prompts):
        for sample_idx in range(num_samples):
            sample_seed = seed + sample_idx
            out_path    = os.path.join(out_dir, f"{_safe(prompt)}-{sample_idx}-{sample_seed}.mp4")

            elapsed = time.time() - t_start
            pct     = 100 * done / total if total else 0
            eta = avg = ""
            if generated > 0:
                avg_s     = ok_total_s / generated
                secs_left = avg_s * (total - done)
                eta       = f"  ETA {_fmt_duration(secs_left)}"
                avg       = f"  avg {avg_s/60:.1f}min/video"
            print(f"\n[vbench] [{done+1}/{total}  {pct:.0f}%{eta}{avg}]  elapsed {_fmt_duration(elapsed)}")
            print(f"[vbench] prompt {task_idx+1}/{len(prompts)}  sample {sample_idx+1}/{num_samples}  seed {sample_seed}")
            print(f"[vbench] {prompt[:70]}")
            if image_path:
                print(f"[vbench] image  {image_path}")

            if os.path.exists(out_path):
                print(f"[vbench] → SKIP (already exists)")
                skipped += 1
                done    += 1
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", "", "", out_path, "skipped"])
                stats_f.flush()
                continue

            try:
                torch.manual_seed(sample_seed)
                st = time.time()

                video_uint8 = generate_video(pipeline, image_path, prompt, num_frames)
                torchvision.io.write_video(out_path, video_uint8, fps=fps)

                duration = time.time() - st
                gen_fps  = num_frames / duration if duration > 0 else 0.0
                _ram_gb  = psutil.Process().memory_info().rss / 1024**3
                _vram_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

                ok_total_s += duration
                print(f"[vbench] saved  {out_path}")
                print(f"[vbench]   {_fmt_duration(duration)}  |  {gen_fps:.1f} gen-fps  |  RAM {_ram_gb:.1f}GB  VRAM {_vram_gb:.1f}GB")
                stats_w.writerow([task_idx, prompt, sample_idx, f"{duration:.2f}", f"{gen_fps:.2f}",
                                  f"{_ram_gb:.2f}", f"{_vram_gb:.2f}", out_path, "ok"])
                stats_f.flush()
                generated += 1

            except Exception as exc:
                print(f"[vbench] ERROR task {task_idx} sample {sample_idx}: {exc}")
                traceback.print_exc()
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", "", "", out_path, "error"])
                stats_f.flush()
                errors += 1

            done += 1

    elapsed_total = time.time() - t_start
    stats_f.close()
    print(f"\n{'='*70}")
    print(f"[vbench] DONE  generated={generated}  skipped={skipped}  errors={errors}")
    print(f"[vbench] total elapsed : {_fmt_duration(elapsed_total)}")
    if generated:
        print(f"[vbench] avg per video : {ok_total_s/generated/60:.1f} min")
    print(f"[vbench] videos → {out_dir}")
    print(f"[vbench] stats  → {stats_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
