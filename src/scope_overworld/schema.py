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

    supports_prompts = True
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
    """Configuration for Waypoint 1.5 360p pipeline (laptop GPUs / Apple Silicon)."""

    pipeline_id = "waypoint_360p"
    pipeline_name = "Waypoint 1.5 (360p)"
    pipeline_description = (
        "Streaming pipeline for Waypoint-1.5-1B-360P (lighter hardware: laptop GPUs, Apple Silicon)."
    )

    artifacts: ClassVar[list[Artifact]] = [
        HuggingfaceRepoArtifact(
            repo_id="Overworld/Waypoint-1.5-1B-360P",
            files=["model.safetensors", "config.yaml"],
        ),
    ]

    height: int = 360
    width: int = 640
