from typing import TYPE_CHECKING, ClassVar

import torch
from torchvision.io import read_image

from scope.core.config import get_model_file_path
from scope.core.pipelines.controller import CtrlInput, convert_to_win_keycodes
from scope.core.pipelines.interface import Pipeline
from world_engine import CtrlInput as WorldCtrlInput
from world_engine import WorldEngine

from .schema import Waypoint360Config, WaypointConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig


class WaypointPipeline(Pipeline):
    """Waypoint 1.5 (720p) pipeline."""

    model_repo_name: ClassVar[str] = "Waypoint-1.5-1B"

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return WaypointConfig

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        quant: str | None = None,
        warmup_iters: int = 2,
        **kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        # Local dir holding model.safetensors + config.yaml (pre-downloaded by Scope's
        # artifact system). world_engine accepts either an HF URI or a local folder.
        model_path = str(get_model_file_path(self.model_repo_name))
        # Point world_engine at the Scope-managed taehv1_5 dir so it uses our
        # artifact cache rather than downloading into ~/.cache/huggingface.
        ae_path = str(get_model_file_path("taehv1_5"))

        # world_engine passes device straight into safetensors.load_file, whose
        # Rust bindings accept only str/int (not torch.device). Pass a string.
        self.engine = WorldEngine(
            model_path,
            quant=quant,
            model_config_overrides={"ae_uri": ae_path},
            device=str(self.device),
            dtype=self.dtype,
        )

        self._warmup(warmup_iters)

    def _warmup(self, n_iters: int) -> None:
        """Run warmup iterations to trigger JIT compilation. Each iter emits 4 frames."""
        for _ in range(n_iters):
            self.engine.gen_frame(ctrl=WorldCtrlInput())

    def __call__(self, **kwargs) -> dict:
        """Generate a 4-frame chunk with controller input.

        Returns:
            dict: {"video": tensor of shape (4, H, W, 3) in [0, 1]}.
            Scope's pipeline processor splits the T=4 output into per-frame packets.
        """
        manage_cache = kwargs.get("manage_cache", False)
        init_cache = kwargs.get("init_cache", False)
        images = kwargs.get("images")

        if manage_cache and images and len(images) > 0:
            init_cache = True

        if init_cache:
            self.engine.reset()

        # Handle image conditioning
        if images and len(images) > 0:
            image = read_image(images[0])  # CHW uint8
            image = image.permute(1, 2, 0)  # HWC uint8
            self.engine.append_frame(image)

        # Controller input: convert Scope's W3C codes to Windows virtual keycodes
        ctrl_input: CtrlInput = kwargs.get("ctrl_input") or CtrlInput()
        win_keys = convert_to_win_keycodes(ctrl_input)
        ctrl = WorldCtrlInput(button=win_keys, mouse=ctrl_input.mouse)

        # Waypoint-1.5 returns (4, H, W, 3) uint8 per call (4x temporal compression).
        frame = self.engine.gen_frame(ctrl=ctrl)
        return {"video": frame.float() / 255.0}


class Waypoint360Pipeline(WaypointPipeline):
    """Waypoint 1.5 360p pipeline (laptop-class NVIDIA GPUs)."""

    model_repo_name: ClassVar[str] = "Waypoint-1.5-1B-360P"

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return Waypoint360Config
