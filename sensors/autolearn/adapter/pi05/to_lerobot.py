"""A behavior_to_lerobot.py dataset (LeRobot v2.1, the flywheel layout) recast for FastWAM (XPolicyLab's policy/FastWAM).

    lerobot_to_fastwam.py <flywheel dataset> --out <fastwam dataset>

FastWAM's loader reads one state and one action vector and three cameras under XPolicyLab's standard names. Here both
vectors are the R1 Pro's arms and torso in joint space, 20 values: left arm (7), left gripper (1, 0 closed .. 1 open),
right arm (7), right gripper (1), torso (4). The torso is in because the demonstrations lean it before reaching, and the
arm targets only fit the leaned posture. Cameras: head -> cam_high, left_wrist -> cam_left_wrist, right_wrist ->
cam_right_wrist; the videos are linked, not re-encoded. Per-episode stats are recomputed for the new keys.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PARTS = ["left_arm", "left_gripper", "right_arm", "right_gripper", "torso"]
CAMERAS = {"head": "cam_high", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
KEEP = ["timestamp", "frame_index", "episode_index", "index", "task_index"]

p = argparse.ArgumentParser()
p.add_argument("src", type=Path)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
src, out = args.src, args.out
info = json.load(open(src / "meta/info.json"))
names = []
for part in PARTS:
    n = info["features"][f"observation.state.{part}"]["shape"][0]
    names += [f"{part}_{i}" for i in range(n)] if n > 1 else [part]
dim = len(names)

if out.exists():
    shutil.rmtree(out)
(out / "meta").mkdir(parents=True)
for f in ("tasks.jsonl", "episodes.jsonl"):
    shutil.copy(src / "meta" / f, out / "meta" / f)

stats_out = open(out / "meta/episodes_stats.jsonl", "w")
size = int(info["chunks_size"])
for line in open(src / "meta/episodes.jsonl"):
    ep = json.loads(line)["episode_index"]
    rel = info["data_path"].format(episode_chunk=ep // size, episode_index=ep)
    t = pq.read_table(src / rel)
    cols = {k: t[k] for k in KEEP}
    stats = {}
    for group in ("observation.state", "action"):
        v = np.concatenate([np.stack(t[f"{group}.{part}"].to_numpy(zero_copy_only=False)).reshape(len(t), -1) for part in PARTS],
                           axis=1).astype(np.float32)
        cols[group] = pa.array(list(v), type=pa.list_(pa.float32(), dim))
        stats[group] = {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(), "std": v.std(0).tolist(),
                        "count": [len(v)]}
    (out / rel).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), out / rel)
    for k in KEEP:
        vals = np.asarray(t[k].to_numpy(), dtype=np.float64)
        stats[k] = {"min": [vals.min()], "max": [vals.max()], "mean": [vals.mean()], "std": [vals.std()], "count": [len(vals)]}
    stats_out.write(json.dumps({"episode_index": ep, "stats": stats}) + "\n")
stats_out.close()
# image stats: the source's, under the new names (the pixels are the same)
img_stats = {}
for line in open(src / "meta/episodes_stats.jsonl"):
    e = json.loads(line)
    img_stats[e["episode_index"]] = {new: e["stats"][f"observation.images.{old}"] for old, new in CAMERAS.items()}
rows = [json.loads(l) for l in open(out / "meta/episodes_stats.jsonl")]
with open(out / "meta/episodes_stats.jsonl", "w") as f:
    for r in rows:
        for new, s in img_stats[r["episode_index"]].items():
            r["stats"][f"observation.images.{new}"] = s
        f.write(json.dumps(r) + "\n")

for chunk in sorted((src / "videos").iterdir()):
    for old, new in CAMERAS.items():
        (out / "videos" / chunk.name).mkdir(parents=True, exist_ok=True)
        os.symlink((chunk / f"observation.images.{old}").resolve(), out / "videos" / chunk.name / f"observation.images.{new}")

feats = {f"observation.images.{new}": info["features"][f"observation.images.{old}"] for old, new in CAMERAS.items()}
feats["observation.state"] = {"dtype": "float32", "shape": [dim], "names": names}
feats["action"] = {"dtype": "float32", "shape": [dim], "names": names}
feats.update({k: info["features"][k] for k in KEEP})
info["features"] = feats
json.dump(info, open(out / "meta/info.json", "w"), indent=2)
print(f"{info['total_episodes']} episodes, {info['total_frames']} frames, state/action {dim}: {names}")
