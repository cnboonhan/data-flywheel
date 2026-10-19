"""Raw episode folders from `run_geniesim.py --record` into one LeRobot v2.1 dataset.

    aspire/.venv-lerobot/bin/python adapter/to_lerobot.py <record dir>/<task> --out aspire/outputs/datasets/<task> [--all]

Only successful episodes by default (`--all` keeps every one). Features follow the Galaxea LeRobot layout the
flywheel's adapter reads (`services/gitea/adapter/galaxeaOpenWorldDataset/`): `observation.images.{head,left_wrist,
right_wrist}` as video, `observation.state.{left_arm,right_arm}` (7 joints each on the G2), `.{left,right}_gripper`
(the benchmark's joint value), `.waist`; `action.*` the same groups from the targets sent; the instruction as the task.
LeRobot 0.3.3 writes v2.1 (`uv venv --python 3.10 aspire/.venv-lerobot && uv pip install lerobot==0.3.3`).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 30                       # the benchmark's control rate (dt = 1/30 s), one recorded frame per tick
CAMERAS = {"head": "head", "hand_left": "left_wrist", "hand_right": "right_wrist"}

p = argparse.ArgumentParser()
p.add_argument("record_dir", type=Path, help="<--record dir>/<task>, holding episode_NNNN/ folders")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--all", action="store_true", help="keep failed episodes too")
p.add_argument("--repo-id", default=None, help="LeRobot repo id (default geniesim/<task>)")
args = p.parse_args()

episodes = sorted(d for d in args.record_dir.iterdir() if d.is_dir() and (d / "frames.jsonl").exists())
kept = [d for d in episodes if args.all or json.loads((d / "episode.json").read_text()).get("success")]
print(f"{len(kept)} of {len(episodes)} episodes" + ("" if args.all else " (successful)"), flush=True)
if not kept:
    raise SystemExit("nothing to convert")

first = json.loads((kept[0] / "frames.jsonl").open().readline())
n_waist = len(first["state"].get("waist_joint_states", []))
shapes = {"left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1}
if n_waist:
    shapes["waist"] = n_waist
features = {}
for cam, name in CAMERAS.items():
    img = next((kept[0] / cam).glob("00000.*"))
    w, h = Image.open(img).size
    features[f"observation.images.{name}"] = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
for group in ("observation.state", "action"):
    for k, n in shapes.items():
        features[f"{group}.{k}"] = {"dtype": "float32", "shape": (n,), "names": [f"{k}_{i}" for i in range(n)]}
for side in ("left", "right"):     # the benchmark's own FK of the gripper centre in base_link, [x y z qx qy qz qw] as Galaxea
    features[f"observation.state.{side}_ee_pose"] = {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]}

if args.out.exists():
    shutil.rmtree(args.out)
ds = LeRobotDataset.create(args.repo_id or f"geniesim/{args.record_dir.name}", FPS, root=args.out, robot_type="agibot_g2_omnipicker",
                           features=features, use_videos=True, image_writer_threads=4)


def groups(state: dict, action: dict) -> tuple[dict, dict]:
    arm = np.asarray(state["arm_joint_states"], dtype=np.float32)
    grip = np.asarray(state.get("gripper_states", [0, 0]), dtype=np.float32)
    s = {"left_arm": arm[:7], "right_arm": arm[7:14], "left_gripper": grip[:1], "right_gripper": grip[1:2]}
    a = {"left_arm": np.asarray(action["left_arm"], dtype=np.float32), "right_arm": np.asarray(action["right_arm"], dtype=np.float32),
         "left_gripper": np.asarray(action["left_effector"], dtype=np.float32), "right_gripper": np.asarray(action["right_effector"], dtype=np.float32)}
    if n_waist:
        s["waist"] = np.asarray(state["waist_joint_states"], dtype=np.float32)
        a["waist"] = np.asarray(action.get("waist", state["waist_joint_states"]), dtype=np.float32)
    return s, a


for d in kept:
    meta = json.loads((d / "episode.json").read_text())
    for line in (d / "frames.jsonl").open():
        fr = json.loads(line)
        s, a = groups(fr["state"], fr["action"])
        frame = {}
        for cam, name in CAMERAS.items():
            img = next((d / cam).glob(f"{fr['frame']:05d}.*"))
            frame[f"observation.images.{name}"] = np.asarray(Image.open(img).convert("RGB"))
        for k in shapes:
            frame[f"observation.state.{k}"] = s[k]
            frame[f"action.{k}"] = a[k]
        for side in ("left", "right"):
            ee = (fr.get("end_pose") or {}).get("base_link", {}).get(f"{side}_arm", {})
            frame[f"observation.state.{side}_ee_pose"] = np.asarray(list(ee.get("position", [0, 0, 0])) + list(ee.get("orientation", [0, 0, 0, 1])), dtype=np.float32)
        ds.add_frame(frame, task=meta["instruction"])
    ds.save_episode()
    print(f"{d.name}: {meta['frames']} frames, {meta['instruction']!r}", flush=True)
print(f"wrote {ds.num_episodes} episodes, {ds.num_frames} frames to {args.out}", flush=True)
