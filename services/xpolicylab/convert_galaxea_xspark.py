#!/usr/bin/env python
"""Convert Galaxea Open-World (LeRobot v2.1, r1lite) episodes into XPolicyLab
"xspark v1.0" HDF5 files, one per episode, for process_data.sh/train.sh.

Input:  <root>/<subset>/            one unpacked Galaxea task dataset
                                     (data/chunk-*/episode_*.parquet, videos/, meta/)
Output: s3://<bucket>/<dest>/<bench>/<subset>/<env_cfg>/data/episode_%07d.hdf5
        (XPolicyLab wants PROJECT_ROOT/data/<bench>/<task>/<env_cfg>/data/)

Layout written (what ACT, DP and the LeRobot converters read):
  data_format_version, instructions (JSON list), subtasks, additional_info/frequency
  state/  left_arm_joint_states (T,6)  left_ee_joint_states (T,1)  right_* likewise
          left_ee_poses (T,7) right_ee_poses (T,7)      [x y z qw qx qy qz]
  action/ same keys; arm/gripper from Galaxea's recorded targets
  vision/cam_head|cam_left_wrist|cam_right_wrist/{colors (T,) S<n> JPEG, shape}

Images are JPEGs stamped with XPolicyLab's XPL-RGB1 marker (the encoder below
mirrors XPolicyLab/utils/process_data.py::encode_image_bit), resized to 480x640.

Needs: h5py pyarrow pandas numpy opencv-python-headless av boto3
"""

import argparse
import glob
import json
import os
import tempfile

import av
import boto3
import cv2
import h5py
import numpy as np
import pandas as pd

CAMERAS = {  # Galaxea video key -> xspark camera group
    "observation.images.head_rgb": "cam_head",
    "observation.images.left_wrist_rgb": "cam_left_wrist",
    "observation.images.right_wrist_rgb": "cam_right_wrist",
}
RESOLUTION = (640, 480)  # (W, H); what RoboDojo data uses and ACT expects

# --- XPolicyLab image encoding (utils/process_data.py) ------------------------
_RGB_MARKER_PAYLOAD = b"XPL-RGB1"
_RGB_MARKER_SEGMENT_LENGTH = len(_RGB_MARKER_PAYLOAD) + 2


def _insert_rgb_marker(jpeg_bytes):
    """Stamp a COM segment after the JFIF APP0 marking the stream as standard RGB."""
    segment = bytes([0xFF, 0xFE]) + _RGB_MARKER_SEGMENT_LENGTH.to_bytes(2, "big") + _RGB_MARKER_PAYLOAD
    offset = 2
    if len(jpeg_bytes) >= 6 and jpeg_bytes[2:4] == bytes([0xFF, 0xE0]):
        app0_end = 4 + int.from_bytes(jpeg_bytes[4:6], "big")
        if app0_end <= len(jpeg_bytes):
            offset = app0_end
    return jpeg_bytes[:offset] + segment + jpeg_bytes[offset:]


def encode_image_bit(rgb, quality=None):
    params = [] if quality is None else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR), params)
    if not ok:
        raise ValueError("JPEG encode failed")
    return _insert_rgb_marker(buf.tobytes())


# --- Galaxea readers ----------------------------------------------------------

def read_frames(path, n):
    """Decode an episode mp4 to n RGB frames at RESOLUTION."""
    frames = []
    with av.open(path) as c:
        for f in c.decode(c.streams.video[0]):
            frames.append(cv2.resize(f.to_ndarray(format="rgb24"), RESOLUTION, interpolation=cv2.INTER_AREA))
            if len(frames) == n:
                break
    if len(frames) < n:
        frames += [frames[-1]] * (n - len(frames))  # pad a short video with its last frame
    return frames


def col(df, name):
    return np.stack(df[name].values).astype(np.float32)


def convert_episode(subset_dir, info, ep, task_of, out_path):
    idx = ep["episode_index"]
    df = pd.read_parquet(f"{subset_dir}/" + info["data_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx))
    T = len(df)
    tasks = [task_of.get(t, str(t)) for t in ep.get("tasks", [])]
    # Galaxea task strings are "<zh>@<en>"; keep the English half as the instruction.
    instructions = [t.split("@", 1)[-1].strip() for t in tasks]

    state = {
        "left_arm_joint_states": col(df, "observation.state.left_arm")[:, :6],
        "left_ee_joint_states": col(df, "observation.state.left_gripper").reshape(T, 1),
        "right_arm_joint_states": col(df, "observation.state.right_arm")[:, :6],
        "right_ee_joint_states": col(df, "observation.state.right_gripper").reshape(T, 1),
    }
    action = {
        "left_arm_joint_states": col(df, "action.left_arm")[:, :6],
        "left_ee_joint_states": col(df, "action.left_gripper").reshape(T, 1),
        "right_arm_joint_states": col(df, "action.right_arm")[:, :6],
        "right_ee_joint_states": col(df, "action.right_gripper").reshape(T, 1),
    }
    # Galaxea poses are [x y z qx qy qz qw]; xspark wants [x y z qw qx qy qz].
    for side in ("left", "right"):
        p = col(df, f"observation.state.{side}_ee_pose")
        state[f"{side}_ee_poses"] = np.concatenate([p[:, :3], p[:, 6:7], p[:, 3:6]], axis=1)
        action[f"{side}_ee_poses"] = np.concatenate([state[f"{side}_ee_poses"][1:], state[f"{side}_ee_poses"][-1:]])

    with h5py.File(out_path, "w") as f:
        s = h5py.string_dtype(encoding="utf-8")
        f.attrs["source_format"] = "galaxea-open-world-r1lite"
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", required=True, help="glob of unpacked Galaxea task datasets under --root, e.g. 'galaxea-open-world-r1lite/Make_The_Bed_*'")
    ap.add_argument("--root", default="/buckets/processed")
    ap.add_argument("--bench", default="Galaxea", help="XPolicyLab bench_name")
    ap.add_argument("--env-cfg", default="r1lite", help="XPolicyLab env_cfg_type (robot)")
    ap.add_argument("--dest", default="xpolicylab", help="prefix in the output bucket")
    ap.add_argument("--bucket", default="processed")
    ap.add_argument("--limit", type=int, default=0, help="episodes per subset (0 = all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    subsets = sorted(d for d in glob.glob(f"{args.root}/{args.subsets}") if os.path.exists(f"{d}/meta/info.json"))
    print(f"{len(subsets)} Galaxea subsets match {args.subsets}")
    for subset_dir in subsets:
        name = os.path.basename(subset_dir.rstrip("/"))
        info = json.load(open(f"{subset_dir}/meta/info.json"))
        task_of = {t["task_index"]: t["task"] for t in map(json.loads, open(f"{subset_dir}/meta/tasks.jsonl"))}
        episodes = [json.loads(l) for l in open(f"{subset_dir}/meta/episodes.jsonl") if l.strip()]
        episodes.sort(key=lambda e: e["episode_index"])
        if args.limit:
            episodes = episodes[: args.limit]
        prefix = f"{args.dest}/{args.bench}/{name}/{args.env_cfg}/data"
        done = {o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=args.bucket, Prefix=prefix + "/") for o in p.get("Contents", [])}
        frames = 0
        # xspark episodes must be numbered contiguously from 0, so use the position,
        # not Galaxea's episode_index (which is contiguous too, but don't rely on it).
        for i, ep in enumerate(episodes):
            key = f"{prefix}/episode_{i:07d}.hdf5"
            if key in done and not args.overwrite:
                continue
            with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmp:
                frames += convert_episode(subset_dir, info, ep, task_of, tmp.name)
                s3.upload_file(tmp.name, args.bucket, key)
        print(f"{name}: {len(episodes)} episodes -> s3://{args.bucket}/{prefix} ({frames} new frames)")


if __name__ == "__main__":
    main()
