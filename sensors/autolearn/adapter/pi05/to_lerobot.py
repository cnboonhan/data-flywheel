"""A behavior_to_lerobot.py dataset (LeRobot v2.1, the flywheel layout) recast for π0.5 (r1pro.py's config) and
converted to LeRobot v3.0, the only version the lerobot 0.4.4 in openpi's lock reads.

    venv-pi05/bin/python to_lerobot.py <flywheel dataset> --out $STATE/pi05/lerobot/r1pro_picking_up_cup

State and action become one joint-space vector of 20 values: left arm (7), left gripper (1, 0 closed .. 1 open), right
arm (7), right gripper (1), torso (4). The torso is in because the demonstrations lean it before reaching, and the arm
targets only fit the leaned posture. Cameras: head -> cam_high, left_wrist -> cam_left_wrist, right_wrist ->
cam_right_wrist; the videos are linked, not re-encoded. The recast v2.1 dataset is written at --out, then lerobot's own
v2.1 -> v3.0 converter (stream copy) replaces it. The output's parent is openpi's HF_LEROBOT_HOME, its name the
config's repo_id.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

PARTS = ["left_arm", "left_gripper", "right_arm", "right_gripper", "torso"]
CAMERAS = {"head": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
KEEP = ["timestamp", "frame_index", "episode_index", "index", "task_index"]

p = argparse.ArgumentParser()
p.add_argument("src", type=Path)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
src, out = args.src.resolve(), args.out
info = json.load(open(src / "meta/info.json"))
names = []
for part in PARTS:
    n = info["features"][f"observation.state.{part}"]["shape"][0]
    names += [f"{part}_{i}" for i in range(n)] if n > 1 else [part]
dim = len(names)

for d in (out, out.with_name(out.name + "_old"), out.with_name(out.name + "_v30")):
    if d.exists():
        shutil.rmtree(d)
(out / "meta").mkdir(parents=True)
for f in ("tasks.jsonl", "episodes.jsonl"):
    shutil.copy(src / "meta" / f, out / "meta" / f)

# per-episode stats: recomputed for the joint vectors and the kept columns; the images' are the source's, renamed
img_stats = {}
for line in open(src / "meta/episodes_stats.jsonl"):
    e = json.loads(line)
    img_stats[e["episode_index"]] = {f"observation.images.{new}": e["stats"][f"observation.images.{old}"] for old, new in CAMERAS.items()}
size = int(info["chunks_size"])
with open(out / "meta/episodes_stats.jsonl", "w") as stats_out:
    for line in open(src / "meta/episodes.jsonl"):
        ep = json.loads(line)["episode_index"]
        rel = info["data_path"].format(episode_chunk=ep // size, episode_index=ep)
        t = pq.read_table(src / rel)
        cols = {k: t[k] for k in KEEP}
        stats = dict(img_stats[ep])
        for group in ("observation.state", "action"):
            v = np.concatenate([np.stack(t[f"{group}.{part}"].to_numpy(zero_copy_only=False)).reshape(len(t), -1) for part in PARTS],
                               axis=1).astype(np.float32)
            cols[group] = pa.array(list(v), type=pa.list_(pa.float32(), dim))
            stats[group] = {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(), "std": v.std(0).tolist(),
                            "count": [len(v)]}
        for k in KEEP:
            vals = np.asarray(t[k].to_numpy(), dtype=np.float64)
            stats[k] = {"min": [vals.min()], "max": [vals.max()], "mean": [vals.mean()], "std": [vals.std()], "count": [len(vals)]}
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(cols), out / rel)
        stats_out.write(json.dumps({"episode_index": ep, "stats": stats}) + "\n")

# one link per video file (not per directory: the converter walks real directories)
for chunk in sorted((src / "videos").iterdir()):
    for old, new in CAMERAS.items():
        d = out / "videos" / chunk.name / f"observation.images.{new}"
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted((chunk / f"observation.images.{old}").iterdir()):
            os.symlink(os.path.realpath(f), d / f.name)

feats = {f"observation.images.{new}": info["features"][f"observation.images.{old}"] for old, new in CAMERAS.items()}
feats["observation.state"] = {"dtype": "float32", "shape": [dim], "names": names}
feats["action"] = {"dtype": "float32", "shape": [dim], "names": names}
feats.update({k: info["features"][k] for k in KEEP})
info["features"] = feats
json.dump(info, open(out / "meta/info.json", "w"), indent=2)
print(f"{info['total_episodes']} episodes, {info['total_frames']} frames, state/action {dim}: {names}")

# the converter moves the v2.1 dataset to <out>_old and writes v3.0 at <out>; the source stays as it is
convert_dataset(repo_id=out.name, root=out.parent, push_to_hub=False, force_conversion=True)
shutil.rmtree(out.with_name(out.name + "_old"))
print(f"LeRobot v3.0 dataset at {out}")
