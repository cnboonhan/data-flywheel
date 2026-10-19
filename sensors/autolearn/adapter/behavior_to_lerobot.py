"""Successful BEHAVIOR-1K runs (ASPIRE's run artifacts, collected by collect_behavior.sh) into one LeRobot v2.1 dataset.

    aspire/.venv-lerobot/bin/python adapter/behavior_to_lerobot.py <collect dir> --out aspire/outputs/datasets/behavior_<suite> [--all]

A run's `episode/steps.jsonl` holds one record per 30 Hz tick (the arm targets sent and the robot state read), and
`media/videos/<camera>.mp4` one frame every third tick (10 fps), so the dataset is at 10 fps with the state of the
tick each frame was taken at. Features follow the Galaxea layout the flywheel's adapter reads, with the R1 Pro's
7-joint arms: `observation.images.{head,left_wrist,right_wrist}`, `observation.state`/`action` `.{left_arm,
right_arm}`, `.{left,right}_gripper` (ASPIRE's 0 closed … 1 open), `observation.state.{left,right}_ee_pose`
([x y z qx qy qz qw] in `odom`), the task language as the task. geniesim_to_xpolicylab.py then writes xspark.
"""

import argparse
import json
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
args = p.parse_args()


def arr(v):
    return np.asarray(v["$array_inline"] if isinstance(v, dict) else v, dtype=np.float32)


runs = sorted(d.parent for d in args.collect_dir.rglob("outcome.json"))
kept = []
for r in runs:
    o = json.loads((r / "outcome.json").read_text())
    if (args.all or o.get("task_success")) and (r / "episode/steps.jsonl").exists() and (r / "media/videos/head.mp4").exists():
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

if args.out.exists():
    shutil.rmtree(args.out)
ds = LeRobotDataset.create(args.repo_id or f"behavior/{args.collect_dir.name}", FPS, root=args.out, robot_type="galaxea_r1pro",
                           features=features, use_videos=True, image_writer_threads=4)

for r in kept:
    task = json.loads((r / "episode/reset.json").read_text())["metadata"]["language"]
    records = [json.loads(l) for l in (r / "episode/steps.jsonl").open() if l.strip()]
    videos = {cam: frames(r / f"media/videos/{cam}.mp4") for cam in CAMERAS}
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
        ds.add_frame(frame, task=task)
    ds.save_episode()
    print(f"{r.parent.parent.name}/{r.parent.name}: {n} frames, {task!r}", flush=True)
print(f"wrote {ds.num_episodes} episodes, {ds.num_frames} frames to {args.out}", flush=True)
