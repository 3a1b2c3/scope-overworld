from typing import TYPE_CHECKING, ClassVar

import torch
from torchvision.io import read_image
from torchvision.transforms.v2.functional import resize as _tv_resize

from scope.core.config import get_model_file_path
from scope.core.pipelines.controller import CtrlInput, convert_to_win_keycodes
from scope.core.pipelines.interface import Pipeline
from world_engine import CtrlInput as WorldCtrlInput
from world_engine import WorldEngine

from .schema import Waypoint1SmallConfig, Waypoint360Config, WaypointConfig

if TYPE_CHECKING:
    from scope.core.pipelines.schema import BasePipelineConfig


class WaypointPipeline(Pipeline):
    """Waypoint 1.5 (720p) pipeline."""

    model_repo_name: ClassVar[str] = "Waypoint-1.5-1B"
    # Canvas the AE expects for this pipeline (taehv1_5 is strictly 16:9).
    canvas_h: ClassVar[int] = 720
    canvas_w: ClassVar[int] = 1280

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
            # Starter-image conditioning (only on cache init). taehv1_5 is a
            # streaming AE with 4x temporal compression that strictly requires
            # 16:9 RGB uint8 input at the pipeline's canvas size, so we
            # center-crop the user's image to 16:9, resize to canvas, and
            # broadcast across the 4-frame temporal window to seed the stream.
            if images and len(images) > 0:
                img = read_image(images[0])  # CHW uint8 at original aspect
                _, h, w = img.shape
                # Center-crop to 16:9
                target = 16 / 9
                if w / h > target:
                    new_w = int(h * target)
                    left = (w - new_w) // 2
                    img = img[:, :, left:left + new_w]
                elif w / h < target:
                    new_h = int(w / target)
                    top = (h - new_h) // 2
                    img = img[:, top:top + new_h, :]
                img = _tv_resize(img, [self.canvas_h, self.canvas_w], antialias=True)
                img = img.permute(1, 2, 0).contiguous()  # HWC uint8
                chunk = img.unsqueeze(0).expand(4, -1, -1, -1).contiguous()
                self.engine.append_frame(chunk)

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
    canvas_h: ClassVar[int] = 360
    canvas_w: ClassVar[int] = 640

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return Waypoint360Config


class Waypoint1SmallPipeline(Pipeline):
    """Waypoint 1 (Small) — the original Waypoint model with text-prompt
    conditioning and an owl_vae autoencoder. Output is a single frame per call.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return Waypoint1SmallConfig

    def __init__(
        self,
        prompt: str = "A fun game",
        n_frames: int = 4096,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        warmup_frames: int = 3,
        **kwargs,
    ):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        model_path = str(get_model_file_path("Waypoint-1-Small"))
        ae_path = str(get_model_file_path("owl_vae_f16_c16_distill_v0_nogan"))
        prompt_encoder_path = str(get_model_file_path("umt5-xl"))

        # world_engine passes device straight into safetensors.load_file, whose
        # Rust bindings accept only str/int (not torch.device). Pass a string.
        self.engine = WorldEngine(
            model_path,
            model_config_overrides={
                "n_frames": n_frames,
                "ae_uri": ae_path,
                "prompt_encoder_uri": prompt_encoder_path,
            },
            device=str(self.device),
            dtype=self.dtype,
        )
        self.engine.set_prompt(prompt)
        self._current_prompt: str | None = prompt

        self._warmup(warmup_frames)

    def _warmup(self, n_frames: int) -> None:
        for _ in range(n_frames):
            self.engine.gen_frame(ctrl=WorldCtrlInput())

    def __call__(self, **kwargs) -> dict:
        """Generate a frame with controller input.

        Returns:
            dict: {"video": tensor of shape (1, H, W, 3) in [0, 1]}.
        """
        manage_cache = kwargs.get("manage_cache", False)
        init_cache = kwargs.get("init_cache", False)
        images = kwargs.get("images")
        prompts = kwargs.get("prompts")

        if manage_cache and images and len(images) > 0:
            init_cache = True

        if init_cache:
            self.engine.reset()

        if images and len(images) > 0:
            image = read_image(images[0])  # CHW uint8
            image = image.permute(1, 2, 0)  # HWC uint8
            self.engine.append_frame(image)

        if prompts and len(prompts) > 0:
            first_prompt = prompts[0]
            new_prompt = (
                first_prompt["text"] if isinstance(first_prompt, dict) else first_prompt
            )
            if new_prompt != self._current_prompt:
                self.engine.set_prompt(new_prompt)
                self._current_prompt = new_prompt

        ctrl_input: CtrlInput = kwargs.get("ctrl_input") or CtrlInput()
        win_keys = convert_to_win_keycodes(ctrl_input)
        ctrl = WorldCtrlInput(button=win_keys, mouse=ctrl_input.mouse)

        frame = self.engine.gen_frame(ctrl=ctrl)
        return {"video": frame.unsqueeze(0).float() / 255.0}
