"""Hooks into IsaacLab-Arena, loaded via PYTHONPATH by envgen.sh. Nothing in the submodule is modified.

Arena's registries are filled by its own modules, and its Streamlit GUI runs in a separate process, so additions are
made by an import hook: right after Arena imports the target module, the addition is registered.

- `cliproxy` inference endpoint (inference_backend): ARENA_PROXY_BASE_URL, ARENA_PROXY_MODEL, key in OPENAI_API_KEY.
- `splat_scene` background (background_library), only when ARENA_SPLAT_SCENE is set: the scene.usda written by
  sim/splat/train.py. The splat stays metric and Z-up; it is shifted in x/y so the centre of its collision floor
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
        """Gaussian splat of a captured scene (sim/splat scene.usda), with its collision floor at z = 0."""

        name = os.getenv("ARENA_SPLAT_NAME", "splat_scene")
        tags = ["background", "splat"]
        usd_path = scene
        initial_pose = Pose(position_xyz=(-cx, -cy, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        object_min_z = -0.2

    register_asset(SplatSceneBackground)


_HOOKS = {"isaaclab_arena.agentic_environment_generation.inference_backend": _register_cliproxy}
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
