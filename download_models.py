"""Download all model files required by WaypointPipeline."""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path(os.environ.get("DAYDREAM_SCOPE_MODELS_DIR",
                                  Path.home() / ".daydream-scope" / "models"))

MODELS = [
    ("Overworld/Waypoint-1-Small", "Waypoint-1-Small", [
        "model.safetensors",
        "config.yaml",
    ]),
    ("OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan", "owl_vae_f16_c16_distill_v0_nogan", [
        "encoder.safetensors",
        "encoder_conf.yml",
        "decoder.safetensors",
        "decoder_conf.yml",
    ]),
    ("google/umt5-xl", "umt5-xl", [
        "config.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
        "pytorch_model.bin.index.json",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ]),
]


def main():
    print(f"Models dir: {MODELS_DIR}\n")
    for repo_id, local_name, files in MODELS:
        dest = MODELS_DIR / local_name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"[{repo_id}]")
        for filename in files:
            out_path = dest / filename
            if out_path.exists():
                print(f"  ✓ {filename} (exists)")
                continue
            print(f"  ↓ {filename} ...", end="", flush=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(dest),
            )
            print(" done")
        print()
    print("All models downloaded.")


if __name__ == "__main__":
    main()
