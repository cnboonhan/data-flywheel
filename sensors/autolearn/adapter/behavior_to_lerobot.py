"""Successful BEHAVIOR-1K runs (ASPIRE's run artifacts, collected by collect_behavior.sh) into one LeRobot v2.1 dataset.

    aspire/.venv-lerobot/bin/python adapter/behavior_to_lerobot.py <collect dir> --out aspire/outputs/datasets/behavior_<suite> [--all]

A run's `episode/steps.jsonl` holds one record per 30 Hz tick (the arm targets sent and the robot state read), and
`media/videos/<camera>.mp4` one frame every third tick (10 fps), so the dataset is at 10 fps with the state of the
tick each frame was taken at. Features follow the Galaxea layout the flywheel's adapter reads, with the R1 Pro's
7-joint arms: `observation.images.{head,left_wrist,right_wrist}`, `observation.state`/`action` `.{left_arm,
right_arm}`, `.{left,right}_gripper` (ASPIRE's 0 closed … 1 open), `observation.state.{left,right}_ee_pose`
([x y z qx qy qz qw] in `odom`), the task language as the task, and the whole body from `episode/whole_body.jsonl`
(run_behavior.py): `observation.state.base_pose` (x, y, yaw in odom), `observation.state.torso`, `action.torso`
(targets), `action.base_velocity` (vx, vy, wz in the base frame, from the next frame's pose). Runs recorded before
that file existed are skipped unless `--arms-only`. geniesim_to_xpolicylab.py then writes xspark.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 10                   # ASPIRE records a BEHAVIOR video frame every third 30 Hz tick
VIDEO_EVERY = 3
CAMERAS = {"head": "head", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
ARMS = {"primary": "left", "secondary": "right"}

p = argparse.ArgumentParser()
p.add_argument("collect_dir", type=Path, help="a collect_behavior.sh output dir (collect.jsonl inside), or any dir of ASPIRE runs")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--all", action="store_true", help="keep failed runs too")
p.add_argument("--repo-id", default=None)
p.add_argument("--format", choices=("flywheel", "galaxea"), default="flywheel",
               help="flywheel: the layout geniesim_to_xpolicylab reads; galaxea: GalaxeaVLA's R1 Pro keys "
                    "(adapter/galaxea/data/r1pro_sim/eef_torso_chassis.yaml)")
p.add_argument("--arms-only", action="store_true",
               help="keep runs recorded before episode/whole_body.jsonl existed (no base or torso features)")
args = p.parse_args()


def arr(v):
    return np.asarray(v["$array_inline"] if isinstance(v, dict) else v, dtype=np.float32)


runs = sorted(Path(dp) for dp, dn, fs in os.walk(args.collect_dir, followlinks=True) if "outcome.json" in fs)   # follows links
kept = []
for r in runs:
    o = json.loads((r / "outcome.json").read_text())
    whole = (r / "episode/whole_body.jsonl").exists()
    if (args.all or o.get("task_success")) and (r / "episode/steps.jsonl").exists() and (r / "media/videos/head.mp4").exists() \
            and (whole or args.arms_only):
        kept.append(r)
print(f"{len(kept)} of {len(runs)} runs" + ("" if args.all else " (successful)"), flush=True)
if not kept:
    raise SystemExit("nothing to convert")


def frames(path):
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


h, w = frames(kept[0] / "media/videos/head.mp4")[0].shape[:2]
features = {f"observation.images.{n}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]} for n in CAMERAS.values()}
for group in ("observation.state", "action"):
    for side in ARMS.values():
        features[f"{group}.{side}_arm"] = {"dtype": "float32", "shape": (7,), "names": [f"{side}_arm_{i}" for i in range(7)]}
        features[f"{group}.{side}_gripper"] = {"dtype": "float32", "shape": (1,), "names": [f"{side}_gripper"]}
for side in ARMS.values():
    features[f"observation.state.{side}_ee_pose"] = {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]}
if not args.arms_only:     # the whole body: base pose in odom, base velocity in the base frame, torso joints
    features["observation.state.base_pose"] = {"dtype": "float32", "shape": (3,), "names": ["x", "y", "yaw"]}
    features["observation.state.torso"] = {"dtype": "float32", "shape": (4,), "names": [f"torso_{i}" for i in range(4)]}
    features["action.torso"] = {"dtype": "float32", "shape": (4,), "names": [f"torso_{i}" for i in range(4)]}
    features["action.base_velocity"] = {"dtype": "float32", "shape": (3,), "names": ["vx", "vy", "wz"]}

GALAXEA = args.format == "galaxea"
if GALAXEA:
    if args.arms_only:
        raise SystemExit("--format galaxea needs the whole-body record (base, torso)")
    vec = lambda n, names: {"dtype": "float32", "shape": (n,), "names": names}
    pose = ["x", "y", "z", "qx", "qy", "qz", "qw"]
    features = {f"observation.images.{n}_rgb": features[f"observation.images.{n}"] for n in ("head", "left_wrist", "right_wrist")}
    for group in ("observation.state", "action"):
        for side in ARMS.values():
            features[f"{group}.{side}_ee_pose"] = vec(7, pose)
            features[f"{group}.{side}_gripper"] = vec(1, [f"{side}_gripper"])
        features[f"{group}.torso"] = vec(4, [f"torso_{i}" for i in range(4)])
    features["action.chassis.velocities"] = vec(6, ["vx", "vy", "vz", "wx", "wy", "wz"])

if args.out.exists():
    shutil.rmtree(args.out)
ds = LeRobotDataset.create(args.repo_id or f"behavior/{args.collect_dir.name}", FPS, root=args.out, robot_type="R1Pro" if GALAXEA else "galaxea_r1pro",
                           features=features, use_videos=True, image_writer_threads=4)

def quat_mul(a, b):   # xyzw
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def in_base(pose_xyzw, base):
    """An odom pose [x y z qx qy qz qw] in the base frame of the base pose (x, y, yaw)."""
    x, y, yaw = base
    dx, dy = pose_xyzw[0] - x, pose_xyzw[1] - y
    c, s_ = math.cos(yaw), math.sin(yaw)
    q = quat_mul([0.0, 0.0, -math.sin(yaw / 2), math.cos(yaw / 2)], list(pose_xyzw[3:]))
    return np.asarray([c * dx + s_ * dy, -s_ * dx + c * dy, pose_xyzw[2]] + q, dtype=np.float32)


def galaxea_frame(f, rec, records, k, body):
    """The flywheel frame recast in GalaxeaVLA's R1 Pro keys: end-effector poses in the base frame (the action is the
    next frame's pose), grippers 0..100, torso joints and targets, chassis velocity as 6 values."""
    base = f["observation.state.base_pose"]
    nxt = f.get("_next_ee") or {}
    g = {}
    for name in ("head", "left_wrist", "right_wrist"):
        g[f"observation.images.{name}_rgb"] = f[f"observation.images.{name}"]
    for side in ARMS.values():
        ee = in_base(f[f"observation.state.{side}_ee_pose"], base)
        g[f"observation.state.{side}_ee_pose"] = ee
        g[f"action.{side}_ee_pose"] = in_base(nxt[side], base) if side in nxt else ee
        g[f"observation.state.{side}_gripper"] = f[f"observation.state.{side}_gripper"] * 100.0
        g[f"action.{side}_gripper"] = f[f"action.{side}_gripper"] * 100.0
    g["observation.state.torso"] = f["observation.state.torso"]
    g["action.torso"] = f["action.torso"]
    vx, vy, wz = f["action.base_velocity"]
    g["action.chassis.velocities"] = np.asarray([vx, vy, 0.0, 0.0, 0.0, wz], dtype=np.float32)
    return g


for r in kept:
    task = json.loads((r / "episode/reset.json").read_text())["metadata"]["language"]
    records = [json.loads(l) for l in (r / "episode/steps.jsonl").open() if l.strip()]
    videos = {cam: frames(r / f"media/videos/{cam}.mp4") for cam in CAMERAS}
    body = {}
    if not args.arms_only:
        for l in (r / "episode/whole_body.jsonl").open():
            row = json.loads(l)
            body[row["tick"]] = row
    n = min(len(v) for v in videos.values())
    if n != -(-len(records) // VIDEO_EVERY):      # not exactly every third tick: spread the frames over the records
        picks = np.linspace(0, len(records) - 1, n).round().astype(int)
    else:
        picks = np.arange(n) * VIDEO_EVERY
    for i, k in enumerate(picks):
        rec = records[min(k, len(records) - 1)]
        st, act = rec["robot_state"], rec["action"]["arms"]
        frame = {}
        for cam, name in CAMERAS.items():
            frame[f"observation.images.{name}"] = videos[cam][i]
        for arm, side in ARMS.items():
            q = arr(st["joint_positions"][arm])
            g = np.asarray([st["gripper_positions"][arm]], dtype=np.float32)
            a = act.get(arm) or {}
            frame[f"observation.state.{side}_arm"] = q
            frame[f"observation.state.{side}_gripper"] = g
            frame[f"action.{side}_arm"] = arr(a["target"]) if a.get("target") is not None else q
            frame[f"action.{side}_gripper"] = np.asarray([a["gripper_position"]], dtype=np.float32) if a.get("gripper_position") is not None else g
            pose = st["end_effector_poses"][arm]
            wxyz = arr(pose["quaternion_wxyz"])
            frame[f"observation.state.{side}_ee_pose"] = np.concatenate([arr(pose["position"]), wxyz[[1, 2, 3, 0]]]).astype(np.float32)
            if GALAXEA:     # the next frame's pose, for the action
                pn = records[min(k + VIDEO_EVERY, len(records) - 1)]["robot_state"]["end_effector_poses"][arm]
                wn = arr(pn["quaternion_wxyz"])
                frame.setdefault("_next_ee", {})[side] = np.concatenate([arr(pn["position"]), wn[[1, 2, 3, 0]]]).astype(np.float32)
        if not args.arms_only:
            tick = int(rec.get("step_index", k + 1))
            b = body.get(tick) or body[min(body, key=lambda t: abs(t - tick))]
            nxt_tick = int(records[min(k + VIDEO_EVERY, len(records) - 1)].get("step_index", tick + VIDEO_EVERY))
            b2 = body.get(nxt_tick) or body[min(body, key=lambda t: abs(t - nxt_tick))]
            x, y, yaw = b["base_pose_odom"]
            x2, y2, yaw2 = b2["base_pose_odom"]
            dt = max(nxt_tick - tick, 1) / 30.0          # the 30 Hz control period
            dx, dy = x2 - x, y2 - y                     # the next frame's motion, expressed in the base frame
            vx, vy = (math.cos(yaw) * dx + math.sin(yaw) * dy) / dt, (-math.sin(yaw) * dx + math.cos(yaw) * dy) / dt
            wz = math.atan2(math.sin(yaw2 - yaw), math.cos(yaw2 - yaw)) / dt
            frame["observation.state.base_pose"] = np.asarray([x, y, yaw], dtype=np.float32)
            frame["observation.state.torso"] = np.asarray(b["torso_q"], dtype=np.float32)
            frame["action.torso"] = np.asarray(b["torso_target"], dtype=np.float32)
            frame["action.base_velocity"] = np.asarray([vx, vy, wz], dtype=np.float32)
        if GALAXEA:
            frame = galaxea_frame(frame, rec, records, k, body)
        ds.add_frame(frame, task=task)
    ds.save_episode()
    print(f"{r.parent.parent.name}/{r.parent.name}: {n} frames, {task!r}", flush=True)
print(f"wrote {ds.num_episodes} episodes, {ds.num_frames} frames to {args.out}", flush=True)
