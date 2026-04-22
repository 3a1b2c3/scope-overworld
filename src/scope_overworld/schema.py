"""Configuration schema for Waypoint pipelines."""

from typing import ClassVar, Literal

from pydantic import Field

from scope.core.pipelines.artifacts import Artifact, HuggingfaceRepoArtifact
from scope.core.pipelines.base_schema import (
    BasePipelineConfig,
    CtrlInput,
    ModeDefaults,
    ui_field_config,
)


class WaypointConfig(BasePipelineConfig):
    """Configuration for Waypoint 1.5 (720p) pipeline."""

    pipeline_id = "waypoint"
    pipeline_name = "Waypoint 1.5"
    pipeline_description = (
        "Streaming pipeline for Waypoint-1.5-1B autoregressive video world models from Overworld."
    )
    docs_url = "https://github.com/Overworldai/world_engine"

    # Waypoint-1.5 configs ship with `prompt_conditioning: null` — the model has
    # no text encoder, so text prompts are not a supported input modality.
    # Inputs are the starter image (via `images`) and keyboard/mouse only.
    supports_prompts: ClassVar[bool] = False
    default_temporal_interpolation_method = None
    default_spatial_interpolation_method = None

    supports_cache_management = True
    supports_quantization: ClassVar[bool] = True

    # Waypoint-1.5 targets 60 fps; latching this lets Scope use the advertised
    # native FPS for playback pacing instead of measured throughput.
    frame_rate: ClassVar[float] = 60.0

    modes = {"text": ModeDefaults(default=True)}

    artifacts: ClassVar[list[Artifact]] = [
        HuggingfaceRepoArtifact(
            repo_id="Overworld/Waypoint-1.5-1B",
            files=["model.safetensors", "config.yaml"],
        ),
        # Autoencoder referenced by the model's config.yaml (`ae_uri`). Declared
        # here so Scope manages the download + cache instead of world_engine
        # silently pulling it into the HF hub cache on first load.
        HuggingfaceRepoArtifact(
            repo_id="Overworld-Models/taehv1_5",
            files=["taehv1_5.pth"],
        ),
    ]

    height: int = 720
    width: int = 1280

    # world_engine supports three quantization tiers beyond Scope's shared enum,
    # so we expose them as a plugin-local Literal.
    quant: Literal["intw8a8", "fp8w8a8", "nvfp4"] | None = Field(
        default=None,
        description="Quantization tier: intw8a8 (30xx+), fp8w8a8 (Ada/Hopper+), nvfp4 (Blackwell).",
        json_schema_extra=ui_field_config(
            component="quantization",
            is_load_param=True,
        ),
    )

    # Controller input support - presence of this field enables controller input capture
    ctrl_input: CtrlInput | None = None

    # Reference images for conditioning (presence enables ImageManager UI)
    images: list[str] | None = Field(
        default=None,
        description="List of reference image paths for conditioning",
    )


class Waypoint360Config(WaypointConfig):
    """Configuration for Waypoint 1.5 360p pipeline (laptop-class NVIDIA GPUs).

    Note: world_engine is CUDA-only; Apple Silicon is not supported via this
    pipeline. Mac users should use Overworld's Biome app (https://over.world/install).
    """

    pipeline_id = "waypoint_360p"
    pipeline_name = "Waypoint 1.5 (360p)"
    pipeline_description = (
        "Streaming pipeline for Waypoint-1.5-1B-360P (lighter variant for laptop-class NVIDIA GPUs)."
    )

    artifacts: ClassVar[list[Artifact]] = [
        HuggingfaceRepoArtifact(
            repo_id="Overworld/Waypoint-1.5-1B-360P",
            files=["model.safetensors", "config.yaml"],
        ),
        HuggingfaceRepoArtifact(
            repo_id="Overworld-Models/taehv1_5",
            files=["taehv1_5.pth"],
        ),
    ]

    height: int = 360
    width: int = 640


class Waypoint1SmallConfig(BasePipelineConfig):
    """Configuration for Waypoint 1 (Small) pipeline — the original Waypoint
    model, with text-prompt conditioning and an owl_vae autoencoder.
    """

    pipeline_id = "waypoint_1_small"
    pipeline_name = "Waypoint 1 (Small)"
    pipeline_description = (
        "Streaming pipeline for the original Waypoint-1-Small autoregressive video world model (text prompts + controller)."
    )
    docs_url = "https://github.com/Overworldai/world_engine"

    supports_prompts = True
    default_temporal_interpolation_method = None
    default_spatial_interpolation_method = None

    supports_cache_management = True

    modes = {"text": ModeDefaults(default=True)}

    artifacts: ClassVar[list[Artifact]] = [
        HuggingfaceRepoArtifact(
            repo_id="Overworld/Waypoint-1-Small",
            files=["model.safetensors", "config.yaml"],
        ),
        HuggingfaceRepoArtifact(
            repo_id="OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan",
            files=[
                "encoder.safetensors",
                "encoder_conf.yml",
                "decoder.safetensors",
                "decoder_conf.yml",
            ],
        ),
        HuggingfaceRepoArtifact(
            repo_id="google/umt5-xl",
            files=[
                "config.json",
                "pytorch_model-00001-of-00002.bin",
                "pytorch_model-00002-of-00002.bin",
                "pytorch_model.bin.index.json",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        ),
    ]

    ctrl_input: CtrlInput | None = None

    images: list[str] | None = Field(
        default=None,
        description="List of reference image paths for conditioning",
    )
