#!/usr/bin/env python
"""Genie Sim recordings (LeRobot v2.1, AgiBot G2) -> XPolicyLab "xspark v1.0" HDF5.

Input:  <raw-dir>/<task>/            one LeRobot v2.1 dataset per task (data/chunk-*/episode_*.parquet, videos/, meta/)
Output: <out>/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5 + manifest.json, where <out> is a local directory or
        s3://<bucket>/<prefix> (the gateway, as galaxeaOpenWorldDataset_to_xpolicylab writes it).

A copy of galaxeaOpenWorldDataset_to_xpolicylab.py with the differences of this source: datasets are directories,
not archives; the G2 arms have 7 joints, so arm states are (T,7) and need an XPolicyLab embodiment with that layout
(env_cfg `agibot_g2`, to be added on the XPolicyLab fork); the gripper is the omnipicker joint value (about -0.785
open, 0 closed); end-effector poses are the benchmark's FK in base_link. Incremental per task through manifest.json
as the Galaxea one. --limit caps the episodes per task.

Layout written (what ACT, DP and the LeRobot converters read):
  data_format_version, instructions (JSON list), subtasks, additional_info/frequency
  state/  left_arm_joint_states (T,7)  left_ee_joint_states (T,1)  right_* likewise
          left_ee_poses (T,7) right_ee_poses (T,7)      [x y z qw qx qy qz]
  action/ same keys; arm/gripper from the targets the adapter sent; poses shifted one frame
  vision/cam_head|cam_left_wrist|cam_right_wrist/{colors (T,) S<n> JPEG, shape}

Needs: h5py pyarrow pandas numpy opencv-python-headless av (boto3 for s3:// output)
"""

import argparse
import fnmatch
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "galaxeaOpenWorldDataset"))
from galaxeaOpenWorldDataset_to_xpolicylab import RESOLUTION, encode_image_bit, read_frames, col  # noqa: E402

CAMERAS = {  # to_lerobot.py video key -> xspark camera group
    "observation.images.head": "cam_head",
    "observation.images.left_wrist": "cam_left_wrist",
    "observation.images.right_wrist": "cam_right_wrist",
}


def convert_episode(subset_dir, info, ep, task_of, out_path):
    idx = ep["episode_index"]
    df = pd.read_parquet(f"{subset_dir}/" + info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx))
    T = len(df)
    instructions = [task_of.get(t, str(t)) for t in ep.get("tasks", [])]
    state, action = {}, {}
    for side in ("left", "right"):
        state[f"{side}_arm_joint_states"] = col(df, f"observation.state.{side}_arm")
        state[f"{side}_ee_joint_states"] = col(df, f"observation.state.{side}_gripper").reshape(T, 1)
        action[f"{side}_arm_joint_states"] = col(df, f"action.{side}_arm")
        action[f"{side}_ee_joint_states"] = col(df, f"action.{side}_gripper").reshape(T, 1)
        p = col(df, f"observation.state.{side}_ee_pose")      # [x y z qx qy qz qw] -> xspark [x y z qw qx qy qz]
        state[f"{side}_ee_poses"] = np.concatenate([p[:, :3], p[:, 6:7], p[:, 3:6]], axis=1)
        action[f"{side}_ee_poses"] = np.concatenate([state[f"{side}_ee_poses"][1:], state[f"{side}_ee_poses"][-1:]])
    with h5py.File(out_path, "w") as f:
        s = h5py.string_dtype(encoding="utf-8")
        f.attrs["source_format"] = "geniesim-lerobot-v2.1"
        f.attrs["source_path"] = f"{os.path.basename(subset_dir)}/episode_{idx:06d}"
        f.create_dataset("data_format_version", data="v1.0", dtype=s)
        f.create_dataset("instructions", data=json.dumps(instructions or [""], ensure_ascii=False), dtype=s)
        f.create_dataset("subtasks", data=json.dumps(instructions, ensure_ascii=False), dtype=s)
        f.create_group("additional_info").create_dataset("frequency", data=np.asarray(info["fps"], dtype=np.int32))
        for group, arrays in (("state", state), ("action", action)):
            g = f.create_group(group)
            for k, v in arrays.items():
                g.create_dataset(k, data=v)
        vision = f.create_group("vision")
        for key, cam in CAMERAS.items():
            path = f"{subset_dir}/" + info["video_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx, video_key=key)
            jpegs = [encode_image_bit(fr) for fr in read_frames(path, T)]
            g = vision.create_group(cam)
            g.create_dataset("colors", data=jpegs, dtype=f"S{max(map(len, jpegs))}")
            g.create_dataset("shape", data=np.asarray((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.int32))
    return T


class Store:
    """manifest.json and episode files under a local directory or an s3:// prefix."""

    def __init__(self, out):
        self.s3 = None
        if out.startswith("s3://"):
            import boto3
            self.bucket, self.prefix = out[5:].split("/", 1)
            self.s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
        else:
            self.root = out

    def get_json(self, rel):
        if self.s3:
            try:
                return json.loads(self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/{rel}")["Body"].read())
            except Exception as e:  # noqa: BLE001  (the gateway answers 404 as a generic ClientError)
                if getattr(e, "response", {}).get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                    return None
                raise
        path = os.path.join(self.root, rel)
        return json.load(open(path)) if os.path.exists(path) else None

    def put_json(self, rel, obj):
        if self.s3:
            self.s3.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{rel}", Body=json.dumps(obj, indent=1).encode())
        else:
            os.makedirs(os.path.dirname(os.path.join(self.root, rel)), exist_ok=True)
            json.dump(obj, open(os.path.join(self.root, rel), "w"), indent=1)

    def put_file(self, local, rel):
        if self.s3:
            self.s3.upload_file(local, self.bucket, f"{self.prefix}/{rel}")
        else:
            os.makedirs(os.path.dirname(os.path.join(self.root, rel)), exist_ok=True)
            os.replace(local, os.path.join(self.root, rel))

    def __str__(self):
        return f"s3://{self.bucket}/{self.prefix}" if self.s3 else self.root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="*", help="glob of task names (dataset directory names)")
    ap.add_argument("--raw-dir", required=True, help="directory holding one LeRobot dataset directory per task")
    ap.add_argument("--out", default="s3://processed/xpolicylab", help="output root: a local directory or s3://bucket/prefix")
    ap.add_argument("--bench", default="geniesim", help="XPolicyLab bench_name; also the folder under the output root")
    ap.add_argument("--env-cfg", default="agibot_g2", help="XPolicyLab env_cfg_type (robot) the data is filed under")
    ap.add_argument("--scratch", default=tempfile.gettempdir())
    ap.add_argument("--limit", type=int, default=0, help="max episodes per task in the output (0 = all)")
    args = ap.parse_args()

    store = Store(args.out)
    tasks = sorted(d for d in os.listdir(args.raw_dir) if os.path.isfile(os.path.join(args.raw_dir, d, "meta/info.json")) and fnmatch.fnmatch(d, args.tasks))
    print(f"{len(tasks)} Genie Sim datasets match {args.tasks}", flush=True)
    for task in tasks:
        subset_dir = os.path.join(args.raw_dir, task)
        info = json.load(open(f"{subset_dir}/meta/info.json"))
        source = {"dataset": task, "episodes": info["total_episodes"], "frames": info["total_frames"]}
        prefix = f"{args.bench}/{task}/{args.env_cfg}"
        man = store.get_json(f"{prefix}/manifest.json") or {"source": None, "episodes": {}, "complete": False}
        if man["source"] == source and (man["complete"] or (args.limit and len(man["episodes"]) >= args.limit)):
            print(f"{task}: unchanged, {len(man['episodes'])} episodes already converted", flush=True)
            continue
        task_of = {t["task_index"]: t["task"] for t in map(json.loads, open(f"{subset_dir}/meta/tasks.jsonl"))}
        episodes = sorted((json.loads(l) for l in open(f"{subset_dir}/meta/episodes.jsonl") if l.strip()), key=lambda e: e["episode_index"])
        todo = [e for e in episodes if str(e["episode_index"]) not in man["episodes"]]
        room = (args.limit - len(man["episodes"])) if args.limit else len(todo)
        todo = todo[: max(room, 0)]
        frames, nxt = 0, (max(man["episodes"].values()) + 1) if man["episodes"] else 0
        for ep in todo:
            with tempfile.NamedTemporaryFile(suffix=".hdf5", dir=args.scratch, delete=False) as tmp:
                frames += convert_episode(subset_dir, info, ep, task_of, tmp.name)
                store.put_file(tmp.name, f"{prefix}/data/episode_{nxt:07d}.hdf5")
            man["episodes"][str(ep["episode_index"])] = nxt
            man["source"], nxt = source, nxt + 1
            store.put_json(f"{prefix}/manifest.json", man)
        man["source"], man["complete"] = source, len(man["episodes"]) >= len(episodes)
        store.put_json(f"{prefix}/manifest.json", man)
        print(f"{task}: +{len(todo)} episodes ({frames} frames), {len(man['episodes'])}/{len(episodes)} in {store}/{prefix}/data", flush=True)


if __name__ == "__main__":
    main()
