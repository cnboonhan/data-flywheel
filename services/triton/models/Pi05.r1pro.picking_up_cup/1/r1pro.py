"""π0.5 (openpi, XPolicyLab's policy/Pi_05) on the Galaxea R1 Pro: transforms and train configs, registered at runtime.

openpi looks configs up in openpi.training.config._CONFIGS_DICT; register() adds the ones here to it, so openpi's own
scripts (train.py, compute_norm_stats.py) and policy loader run them without changes to the submodule.

The data is to_lerobot.py's: one 20-value joint vector for state and action, left arm (7), left gripper (1, 0 closed
.. 1 open), right arm (7), right gripper (1), torso (4), and three cameras cam_high, cam_left_wrist, cam_right_wrist.
The model sees them as π0.5's base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, the same slots as Aloha's.
"""
import dataclasses
import os
import pathlib

import einops
import numpy as np
import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from typing_extensions import override

STATE = pathlib.Path(os.environ.get("STATE", "."))   # holds pi05/ (norm stats, data) and models/openpi/ (base weights)
PI05 = STATE / "pi05"
# openpi's own download of gs://openpi-assets/checkpoints/pi05_base (setup_pi05.sh)
BASE_PARAMS = STATE / "models/openpi/openpi-assets/checkpoints/pi05_base/params"
DIM = 20
CAMERAS = {"cam_high": "base_0_rgb", "cam_left_wrist": "left_wrist_0_rgb", "cam_right_wrist": "right_wrist_0_rgb"}
PROMPT = "pick up the cup"


def _hwc_uint8(img):
    """LeRobot frames are float CHW in 0..1; the server sends HWC uint8. π0.5 takes HWC uint8."""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.shape[-3] == 3 and img.shape[-1] != 3:
        img = einops.rearrange(img, "... c h w -> ... h w c")
    return img


@dataclasses.dataclass(frozen=True)
class R1ProInputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        images = {dst: _hwc_uint8(data["images"][src]) for src, dst in CAMERAS.items()}
        out = {"image": images, "image_mask": {k: np.True_ for k in images}, "state": np.asarray(data["state"])}
        if "actions" in data:
            out["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class R1ProOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :DIM]}    # the model pads to its action_dim (32)


@dataclasses.dataclass(frozen=True)
class LeRobotR1ProDataConfig(_config.DataConfigFactory):
    default_prompt: str | None = PROMPT

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        data_transforms = _transforms.Group(inputs=[R1ProInputs()], outputs=[R1ProOutputs()])
        # joints as deltas from the current state, grippers absolute, as openpi's Aloha config; the torso is joints too
        mask = _transforms.make_bool_mask(7, -1, 7, -1, 4)
        data_transforms = data_transforms.push(inputs=[_transforms.DeltaActions(mask)],
                                               outputs=[_transforms.AbsoluteActions(mask)])
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "images": {k: f"observation.images.{k}" for k in CAMERAS},
            "state": "observation.state", "actions": "action", "prompt": "prompt"})])
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack, data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config),
            action_sequence_keys=("action",))


def _train_config(name, **kw):
    return _config.TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(pi05=True),   # action_horizon 50 (5 s at the data's 10 fps), openpi's default
        data=LeRobotR1ProDataConfig(repo_id="r1pro_picking_up_cup",
                                    base_config=_config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(BASE_PARAMS)),
        assets_base_dir=str(PI05 / "assets"),
        checkpoint_base_dir=str(PI05 / "checkpoints"),
        wandb_enabled=False,
        **kw)


CONFIGS = [
    # untuned beyond fitting the data: 128 x 20k steps is ~30 passes over the 84,901 frames; openpi's fine-tuning
    # examples run 30k steps at 32, XPolicyLab's RoboDojo config 60k at 256 on far more data
    _train_config("pi05_r1pro_picking_up_cup", batch_size=128, num_train_steps=20_000, save_interval=2_000,
                  keep_period=5_000, num_workers=32),   # 32 loader workers: AV1 decoding of 3 cameras per frame is the bottleneck
]


def register():
    for c in CONFIGS:
        _config._CONFIGS_DICT[c.name] = c
