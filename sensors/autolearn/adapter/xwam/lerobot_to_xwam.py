"""A behavior_to_lerobot.py dataset (LeRobot v2.1, the flywheel layout) recast for X-WAM (XPolicyLab's policy/X_WAM).

    venv-xwam/bin/python lerobot_to_xwam.py <flywheel dataset> --out <xwam dataset> [--workers 32]

X-WAM reads one JSON per episode (proprios: both hands' end-effector position, rotation matrix and gripper; actions: the
next frame's pose, from which its loader takes per-step deltas) and one video per camera, the layout XPolicyLab's
transform_robodojo_to_xwam.py writes. Here:

- poses are in the base frame (behavior_to_lerobot.py's in_base, as the GalaxeaVLA data and eval_galaxea_behavior.py);
- each hand's rotation is redefined by a fixed rotation, R @ P with P the transpose of that hand's mean orientation
  over the data (written to frame.json; serve_xwam.py applies the same), as upstream's transform_robodojo_to_xwam.py
  does with its own P for the arx: X-WAM feeds the proprio rotation as a quaternion with w >= 0, and the R1 Pro's
  grasp orientations sit near w = 0 (median |w| 0.10), where that form flips between q and -q from frame to frame;
  reframed, the median |w| is 0.98. The actions are unaffected: X-WAM's rotation deltas a_R p_R^T cancel P;
- grippers are ASPIRE's 0 closed .. 1 open;
- each episode starts where the base stops for good (the eval starts parked at the counter, at the scripted run's
  base_after_hop): the search and the drive before it hold the base-frame poses still and would teach standing still;
- head -> head_camera (static), left_wrist -> left_camera, right_wrist -> right_camera (dynamic); videos re-encoded to
  H.264 at VIDEO_SIZE (X-WAM resizes to its video_size anyway; small clips decode fast through the PyAV decord shim).

X-WAM has no torso output: the torso lean the demonstrations make before reaching is left to serve_xwam.py, which holds
the torso at the lean this script records in torso.json (the median of each episode's final torso).
Then the quantile statistics: run X-WAM's own scripts/compute_stats.py on the output (--output <out>/quantile.yaml),
which train_xwam.sbatch reads.
"""
import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

CAMERAS = {"head": ("head_camera", "static"), "left_wrist": ("left_camera", "dynamic"),
           "right_wrist": ("right_camera", "dynamic")}
VIDEO_SIZE = 256        # px, square as the 512x512 cameras; X-WAM's video_size is set to match (train_xwam.sbatch)
CRF = 18                # H.264 quality, as transform_robodojo_to_xwam.py (visually lossless)
BASE_STILL_M = 0.001    # base motion per frame (m) below which the base counts as stopped: above the pose noise
BASE_STILL_RAD = 0.002  # (rad) at rest, below any servo step (behavior's hops move cm per frame)
BASE_MIN_RUN = 3        # frames: a drive or hop moves the base for this many frames in a row; shorter runs are contact
                        # jitter at the grasp (seen as single-frame blips) and must not cut the reach out
MIN_FRAMES = 40         # episodes shorter than this after the trim are dropped: X-WAM needs (9 - 1) x 4 + 1 = 33 frames


def col(t, k):
    return np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in t[k].to_pylist()])


def in_base(pose_xyzw, base):
    """odom poses [T, 7] (x y z qx qy qz qw) in the base frame of base poses [T, 3] (x, y, yaw): position, rotation."""
    x, y, yaw = base[:, 0], base[:, 1], base[:, 2]
    dx, dy = pose_xyzw[:, 0] - x, pose_xyzw[:, 1] - y
    c, s = np.cos(yaw), np.sin(yaw)
    pos = np.stack([c * dx + s * dy, -s * dx + c * dy, pose_xyzw[:, 2]], axis=1)
    rot = Rotation.from_euler("z", -yaw) * Rotation.from_quat(pose_xyzw[:, 3:7])
    return pos, rot.as_matrix().reshape(-1, 9)


def frames(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def write_video(rgb, path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=int(round(fps)))
        s.width = s.height = VIDEO_SIZE
        s.pix_fmt = "yuv420p"
        s.options = {"crf": str(CRF)}
        for img in rgb:
            f = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(width=VIDEO_SIZE, height=VIDEO_SIZE)
            for pkt in s.encode(f):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def trim_start(base):
    """The first frame after the base's last drive (BASE_MIN_RUN frames or more of motion)."""
    step = np.abs(np.diff(base, axis=0))
    step[:, 2] = np.abs((np.diff(base[:, 2]) + math.pi) % (2 * math.pi) - math.pi)
    moving = np.nonzero((np.hypot(step[:, 0], step[:, 1]) > BASE_STILL_M) | (step[:, 2] > BASE_STILL_RAD))[0]
    runs = [r for r in np.split(moving, np.nonzero(np.diff(moving) > 1)[0] + 1) if len(r) >= BASE_MIN_RUN]
    return int(runs[-1][-1]) + 1 if runs else 0      # step k moves frame k to k+1: k+1 is the first frame at rest


def rotations(job):
    """The pre-pass: each hand's base-frame rotations over the kept part of an episode, every 5th frame."""
    src, _, ep, _, info, _, _ = job
    size = int(info["chunks_size"])
    t = pq.read_table(src / info["data_path"].format(episode_chunk=ep // size, episode_index=ep))
    base = col(t, "observation.state.base_pose")
    start = trim_start(base)
    if len(t) - start < MIN_FRAMES:
        return None
    return {side: in_base(col(t, f"observation.state.{side}_ee_pose")[start::5], base[start::5])[1] for side in ("left", "right")}


def convert(job):
    src, out, ep, index, info, task, frame = job
    size = int(info["chunks_size"])
    t = pq.read_table(src / info["data_path"].format(episode_chunk=ep // size, episode_index=ep))
    base = col(t, "observation.state.base_pose")
    start = trim_start(base)
    n = len(t) - start
    if n < MIN_FRAMES:
        return {"episode": ep, "kept": False, "frames": n}
    fps = float(info["fps"])
    chunk, name = f"chunk-{index // 1000:04d}", f"episode_{index:07d}"
    observations = {}
    for cam, (field, kind) in CAMERAS.items():
        rgb = frames(src / info["video_path"].format(episode_chunk=ep // size, video_key=f"observation.images.{cam}",
                                                     episode_index=ep))[start:start + n]
        if len(rgb) != n:
            raise ValueError(f"episode {ep} {cam}: {len(rgb)} frames after the trim, {n} rows")
        rel = f"video/{field}/{chunk}/{name}.mp4"
        write_video(rgb, out / rel, fps)
        observations[field] = {"type": kind, "rgb_path": rel, "depth_path": rel, "start": 0, "end": n, "fps": fps}
    proprios = {}
    for side in ("left", "right"):
        pos, rotm = in_base(col(t, f"observation.state.{side}_ee_pose")[start:], base[start:])
        proprios[f"{side}_ee_pos"] = pos
        proprios[f"{side}_ee_rotm"] = (rotm.reshape(-1, 3, 3) @ np.asarray(frame[side])).reshape(-1, 9)
        proprios[f"{side}_gripper_pos"] = col(t, f"observation.state.{side}_gripper")[start:]
    episode = {"num_frames": n, "instructions": [task],
               "observations": observations,
               "proprios": {k: v.tolist() for k, v in proprios.items()},
               "actions": {k: v[1:].tolist() for k, v in proprios.items()}}    # the next frame's pose, as upstream
    (out / "data" / chunk).mkdir(parents=True, exist_ok=True)
    json.dump(episode, open(out / "data" / chunk / f"{name}.json", "w"))
    return {"episode": ep, "kept": True, "frames": n, "start": start, "torso": col(t, "observation.state.torso")[-1].tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()
    src, out = args.src.resolve(), args.out.resolve()
    info = json.load(open(src / "meta/info.json"))
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(src / "meta/tasks.jsonl")}
    eps = [json.loads(l) for l in open(src / "meta/episodes.jsonl")]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    jobs = [(src, out, e["episode_index"], i, info, (e.get("tasks") or [tasks[0]])[0], None) for i, e in enumerate(eps)]
    with ProcessPoolExecutor(args.workers) as ex:
        rots = [r for r in ex.map(rotations, jobs) if r is not None]
        frame = {side: Rotation.from_matrix(np.concatenate([r[side] for r in rots]).reshape(-1, 3, 3)).mean().as_matrix().T.tolist()
                 for side in ("left", "right")}
        json.dump({**frame, "note": "R @ P per hand, P = the transpose of the hand's mean base-frame orientation"},
                  open(out / "frame.json", "w"), indent=1)
        res = list(ex.map(convert, [j[:-1] + (frame,) for j in jobs]))
    kept = [r for r in res if r["kept"]]
    torso = np.median(np.asarray([r["torso"] for r in kept]), axis=0)
    json.dump({"torso": torso.tolist(), "note": "median final torso of the kept episodes; serve_xwam.py holds it"},
              open(out / "torso.json", "w"), indent=1)
    lens = [r["frames"] for r in kept]
    print(f"{len(kept)} of {len(res)} episodes kept, frames per episode {min(lens)}..{max(lens)} "
          f"(median {int(np.median(lens))}), {sum(lens)} frames; torso held at {np.round(torso, 3).tolist()}")


if __name__ == "__main__":
    main()
