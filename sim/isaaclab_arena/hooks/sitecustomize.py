"""Hooks into IsaacLab-Arena, loaded via PYTHONPATH by envgen.sh. Nothing in the submodule is modified.

Arena's registries are filled by its own modules, and its Streamlit GUI runs in a separate process, so additions are
made by an import hook: right after Arena imports the target module, the addition is registered.

- `cliproxy` inference endpoint (inference_backend): ARENA_PROXY_BASE_URL, ARENA_PROXY_MODEL, key in OPENAI_API_KEY.
- `ridgeback_franka_ik` embodiment (franka module): franka_ik on Isaac Lab's holonomic Ridgeback base. Actions:
  franka_ik's 7, then base vx, vy, wz on the planar joints anchored at the spawn pose (so in that frame).
- `splat_scene` background (background_library), only when ARENA_SPLAT_SCENE is set: the scene.usda written by
  sim/colmap_splat/train.py. The splat stays metric and Z-up; it is shifted in x/y so the centre of its collision floor
  (the capture area) is Arena's origin, where embodiments spawn. Reference the floor as `prim_path: floor` to place
  objects on it.
"""

import importlib.abc
import importlib.util
import os
import re
import sys


def _register_cliproxy(module) -> None:
    module.INFERENCE_ENDPOINTS["cliproxy"] = module.InferenceEndpoint(
        name="cliproxy",
        base_url=os.getenv("ARENA_PROXY_BASE_URL", "http://127.0.0.1:8317/v1"),
        model=os.getenv("ARENA_PROXY_MODEL", "claude-sonnet-5-5"),
        api_key_env_var="OPENAI_API_KEY",
    )


def _register_splat(module) -> None:
    scene = os.path.abspath(os.environ["ARENA_SPLAT_SCENE"])
    floor = re.search(r'def Cube "floor".*?xformOp:translate = \(([^)]*)\)', open(scene).read(), re.S)
    cx, cy = (float(v) for v in floor.group(1).split(",")[:2]) if floor else (0.0, 0.0)

    from isaaclab_arena.assets.register import register_asset
    from isaaclab_arena.utils.pose import Pose

    class SplatSceneBackground(module.LibraryBackground):
        """Gaussian splat of a captured scene (sim/colmap_splat scene.usda), with its collision floor at z = 0."""

        name = os.getenv("ARENA_SPLAT_NAME", "splat_scene")
        tags = ["background", "splat"]
        usd_path = scene
        initial_pose = Pose(position_xyz=(-cx, -cy, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        object_min_z = -0.2

    register_asset(SplatSceneBackground)


def _register_ridgeback_franka(module) -> None:
    import isaaclab.sim as sim_utils
    from isaaclab.envs.mdp.actions.actions_cfg import JointVelocityActionCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_assets.robots.ridgeback_franka import RIDGEBACK_FRANKA_PANDA_CFG

    from isaaclab_arena.assets.register import register_asset

    @configclass
    class RidgebackFrankaIKActionCfg(module.FrankaIKActionCfg):
        # Base velocity on the planar joints that carry the base: vx, vy (m/s), wz (rad/s).
        base_action = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint"],
            preserve_order=True,
        )

    class RidgebackFrankaIKEmbodiment(module.FrankaIKEmbodiment):
        """Franka on a holonomic Clearpath Ridgeback: franka_ik arm and gripper, then base velocity."""

        name = "ridgeback_franka_ik"
        tags = ["embodiment", "mobile"]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            robot = self.scene_config.robot   # the franka_ik arm pose and gains, on the Ridgeback instead of a stand
            cfg = RIDGEBACK_FRANKA_PANDA_CFG.replace(
                prim_path=robot.prim_path,
                init_state=RIDGEBACK_FRANKA_PANDA_CFG.init_state.replace(
                    joint_pos={**RIDGEBACK_FRANKA_PANDA_CFG.init_state.joint_pos, **robot.init_state.joint_pos}
                ),
            )
            cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)   # as franka_ik: IK holds pose
            for group in ("panda_shoulder", "panda_forearm"):
                cfg.actuators[group] = cfg.actuators[group].replace(
                    stiffness=robot.actuators[group].stiffness, damping=robot.actuators[group].damping
                )
            self.scene_config.robot = cfg
            self.action_config = RidgebackFrankaIKActionCfg()

    register_asset(RidgebackFrankaIKEmbodiment)


_HOOKS = {
    "isaaclab_arena.agentic_environment_generation.inference_backend": _register_cliproxy,
    "isaaclab_arena.embodiments.franka.franka": _register_ridgeback_franka,
}
if os.getenv("ARENA_SPLAT_SCENE"):
    _HOOKS["isaaclab_arena.assets.background_library"] = _register_splat


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in _HOOKS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            _HOOKS[fullname](module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, _Finder())
