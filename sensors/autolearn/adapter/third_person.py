"""A third-person camera in every BEHAVIOR observation, for videos only: ASPIRE's run observer writes each camera of
the observation to media/videos/<name>.mp4, so the extra one lands beside head.mp4 and the wrist videos, and
behavior_to_lerobot.py (which converts only its three named cameras) copies it to the dataset's visuals/ folder
rather than into a feature. The frame comes from OmniGibson's viewer camera, which renders every step anyway.

    BEHAVIOR_THIRD_PERSON=follow    behind and to the left of the robot, following it (any task)
    BEHAVIOR_THIRD_PERSON=door      a fixed camera beside the doorway, from the door task's frame (run_behavior.py's
                                    DOOR_TASKS); falls back to follow where a task has no door frame

run_behavior.py installs it on BehaviorAdapter.get_observation.
"""
import dataclasses
import math
import os

import numpy as np

NAME = "third_person"
# follow camera, metres from the robot's base: behind it, to its left, up; it looks at the robot's chest (CHEST_Z up,
# a little ahead). 2.2 m back keeps a 2.3 m tall R1 Pro with its arms in a 1280x720 frame; untuned beyond that
FOLLOW = {"back": 2.2, "left": 1.0, "height": 1.7, "chest_z": 1.0, "ahead": 0.5}
# door camera, from the task's door frame (face: the swing-side face at the free edge; swing: the way the leaf opens;
# along: hinge to free edge): out in front of the face on the swing side and past the free edge, so the leaf, the
# lever and the robot in profile are all in view; it looks between the lever (0.15 m in from the free edge, 1.0 m up)
# and the robot's start, `toward_robot` out from the face. At out 1.6 / past 2.4 the wall filled half the frame and
# the robot sat at its edge; untuned beyond this
DOOR = {"out": 2.6, "past": 2.0, "height": 1.9, "lever_in": 0.15, "lever_z": 1.0, "toward_robot": 0.4}

_fixed = None      # (position, target) in world, set by run_behavior.py for door tasks; None follows the robot
_ready = False


def set_fixed(position, target):
    global _fixed
    _fixed = (np.asarray(position, dtype=np.float32), np.asarray(target, dtype=np.float32))


def clear_fixed():
    global _fixed
    _fixed = None


def door_camera(frame):
    """Position and target from a DOOR_TASKS frame (2-D face, swing, along; the door stands on z=0)."""
    f, s, a = (np.asarray(frame[k], dtype=np.float32) for k in ("face", "swing", "along"))
    pos = f + s * DOOR["out"] + a * DOOR["past"]
    look = f - a * DOOR["lever_in"] + s * DOOR["toward_robot"]
    return (float(pos[0]), float(pos[1]), DOOR["height"]), (float(look[0]), float(look[1]), DOOR["lever_z"])


def _look_at(pos, target):
    """Quaternion (x y z w) of a USD camera at pos looking at target: it looks down its -Z, with +Y up."""
    import omnigibson.utils.transform_utils as T
    import torch as th

    z = th.tensor(pos) - th.tensor(target)
    z = z / z.norm()
    x = th.linalg.cross(th.tensor([0.0, 0.0, 1.0]), z)
    x = x / x.norm()
    return T.mat2quat(th.stack([x, th.linalg.cross(z, x), z], dim=1))


def _pose(adapter):
    import omnigibson.utils.transform_utils as T
    import torch as th

    if _fixed is not None:
        return _fixed
    p, q = adapter.robot.get_position_orientation()
    heading = T.quat2mat(q)[:, 0]
    left = th.tensor([-heading[1], heading[0], 0.0])
    pos = p - heading * FOLLOW["back"] + left * FOLLOW["left"] + th.tensor([0.0, 0.0, FOLLOW["height"]])
    look = p + heading * FOLLOW["ahead"] + th.tensor([0.0, 0.0, FOLLOW["chest_z"]])
    return pos.numpy(), look.numpy()


def observation(adapter):
    """A CameraObservation of the viewer camera, placed for this step."""
    global _ready
    import torch as th
    from cap_harness.behavior import adapter as ad
    from cap_harness.contracts import CameraObservation

    cam = adapter._og.sim.viewer_camera
    if not _ready:
        cam.add_modality("rgb")
        _ready = True
    pos, look = _pose(adapter)
    cam.set_position_orientation(th.tensor(pos, dtype=th.float32), _look_at(list(map(float, pos)), list(map(float, look))))
    obs = cam.get_obs()
    obs = obs[0] if isinstance(obs, tuple) else obs
    rgb = obs["rgb"]
    rgb = rgb.cpu().numpy() if hasattr(rgb, "cpu") else np.asarray(rgb)
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))
    world_from_cam = ad._pose_matrix(*cam.get_position_orientation())
    k = cam.intrinsic_matrix
    k = k.cpu().numpy() if hasattr(k, "cpu") else np.asarray(k)
    return CameraObservation(
        rgb=rgb,
        depth_m=np.zeros(rgb.shape[:2], dtype=np.float64),      # no depth: the stream is for watching, not perception
        intrinsics=np.asarray(k, dtype=np.float64),
        frame=f"camera/{NAME}",
        camera_pose=ad.matrix_to_pose(adapter._odom_from_world @ world_from_cam @ ad.GL_TO_CV, frame=adapter.BASE_FRAME),
        timestamp_s=adapter.current_time_s,
    )


def install(adapter_cls):
    """Add the camera to every observation the adapter returns."""
    get = adapter_cls.get_observation

    def get_observation(self, *a, **k):
        obs = get(self, *a, **k)
        try:
            cam = observation(self)
        except Exception as exc:  # the stream must never break a run
            import sys

            print(f"third-person frame skipped: {exc}", file=sys.stderr)
            return obs
        return dataclasses.replace(obs, cameras={**obs.cameras, NAME: cam})

    adapter_cls.get_observation = get_observation


def mode():
    return os.environ.get("BEHAVIOR_THIRD_PERSON", "").strip().lower() or None


def enabled():
    return mode() in ("follow", "door")
