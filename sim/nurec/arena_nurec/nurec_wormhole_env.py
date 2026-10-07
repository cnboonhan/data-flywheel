"""NuRec sample scene as an IsaacLab-Arena background, plus a smoke-test environment around it.

Loaded by policy_runner through `--external_environment_class_path` (see sim/nurec/arena_smoke_test.sh), so the
Arena submodule stays untouched. The scene is the `nova_carter-wormhole` conference room from
nvidia/PhysicalAI-Robotics-NuRec (download with scripts/download.sh): a Gaussian splat
(`ParticleField3DGaussianSplat`) with an aligned collision mesh already carrying `PhysicsMeshCollisionAPI`.
The floor is at z = 0 and the capture trajectory starts at the origin, so rendering is best near it.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.envs.common import ViewerCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.franka.franka import FrankaCameraCfg
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.cube_goal_pose_environment import CubeGoalPoseEnvironment, CubeGoalPoseEnvironmentCfg

ROOT = Path(__file__).resolve().parents[3]
SCENE_DIR = ROOT / "datasets" / "nurec-nova_carter-wormhole" / "nova_carter-wormhole"


def _latest_export(run_glob: str) -> str:
    """Newest 3DGRUT export matching sim/nurec/runs/<run_glob>, or "" if none has been trained yet."""
    hits = sorted((ROOT / "sim" / "nurec" / "runs").glob(f"{run_glob}/*/export_last_lightfield.usdz"))
    return str(hits[-1]) if hits else ""


@register_asset
class NurecNovaCarterWormhole(LibraryBackground):
    """Conference room reconstructed with NuRec's stereo workflow (splat + collision mesh)."""

    name = "nurec_nova_carter_wormhole"
    tags = ["background", "nurec"]
    # Plain (non-PPISP) particle stage: renders without extra carb overrides.
    usd_path = str(SCENE_DIR / "particle_sh_optimized.usdz")
    object_min_z = -0.3

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.8, -2.2, 1.3), lookat=(-0.1, 0.0, 0.3))


@configclass
class NurecFrankaCameraCfg(FrankaCameraCfg):
    """Franka's wrist camera plus a fixed room camera, so the recorded video shows the splat around the robot."""

    scene_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/scene_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955),
        # At (1.8, -2.2, 1.3) looking at (-0.1, 0, 0.3), the robot base. Quaternion is (x, y, z, w).
        offset=CameraCfg.OffsetCfg(
            pos=(1.8, -2.2, 1.3), rot=(-0.1499, 0.0686, 0.8968, 0.4105), convention="world"
        ),
    )


@register_asset
class NurecZhLoungeOurs(LibraryBackground):
    """Our own 3DGRUT reconstruction of NVIDIA's zh_lounge photos (sim/nurec/train.sh). Mono workflow: splat
    only, no collision mesh. 3DGRUT exports Y-up, so rotate +90 deg about X into Isaac Sim's Z-up frame."""

    name = "nurec_zh_lounge_ours"
    tags = ["background", "nurec"]
    usd_path = os.environ.get("NUREC_ZH_LOUNGE_USD") or _latest_export("zh_lounge*")
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.7071068, 0.0, 0.0, 0.7071068))
    object_min_z = -5.0  # nothing to land on; don't reset the cube as soon as it falls

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(1.8, -2.2, 1.3), lookat=(-0.1, 0.0, 0.3))


@dataclass
class NurecWormholeEnvironmentCfg(CubeGoalPoseEnvironmentCfg):
    background: str = "nurec_nova_carter_wormhole"


class NurecWormholeEnvironment(ArenaEnvironmentFactory[NurecWormholeEnvironmentCfg]):
    """cube_goal_pose (Franka + cube on the floor at the origin), inside the NuRec conference room."""

    name = "nurec_wormhole"
    _legacy_argparse_cfg_type = NurecWormholeEnvironmentCfg

    def build(self, cfg: NurecWormholeEnvironmentCfg):
        env = CubeGoalPoseEnvironment.build(self, cfg)
        # Short episodes: the camera video recorder flushes at episode reset, not after N steps.
        env.task.episode_length_s = 3.0
        if cfg.enable_cameras and cfg.embodiment == "franka_ik":
            env.embodiment.camera_config = NurecFrankaCameraCfg()
        return env
