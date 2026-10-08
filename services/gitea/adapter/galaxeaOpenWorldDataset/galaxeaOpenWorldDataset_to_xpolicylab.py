#!/usr/bin/env python
"""Galaxea Open-World (LeRobot v2.1, r1lite) -> XPolicyLab "xspark v1.0" HDF5, incrementally.

Input:  <raw-dir>/<task>.tar.gz     one Galaxea task archive (data/chunk-*/episode_*.parquet, videos/, meta/)
Output: s3://processed/xpolicylab/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5
        + manifest.json next to data/: the source archive (size, mtime) and source episode -> output index.

Incremental: an archive that is unchanged and fully converted is skipped without unpacking. Otherwise it is
unpacked to scratch, and only episodes missing from the manifest are converted, appended after the highest
output index (XPolicyLab wants contiguous numbering from 0, so existing files never move). The manifest is
written after every episode, so an interrupted run resumes. --limit caps the episodes per task.

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
import fnmatch
import json
import os
import shutil
import tarfile
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


def load_manifest(s3, bucket, key):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:  # noqa: BLE001  (gateway answers 404 as a generic ClientError)
        if getattr(e, "response", {}).get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True, help="glob of task names (archive names without .tar.gz), e.g. 'Arrange_Fruits_*' or '*'")
    ap.add_argument("--raw-dir", required=True, help="directory holding the <task>.tar.gz archives (the raw bucket on disk)")
    ap.add_argument("--scratch", default=tempfile.gettempdir(), help="where an archive is unpacked while it converts")
    ap.add_argument("--bench", default="galaxeaOpenWorldDataset", help="XPolicyLab bench_name; also the folder under xpolicylab/")
    ap.add_argument("--env-cfg", default="arx_x5", help="XPolicyLab env_cfg_type (robot); arx_x5 has the same 14-D layout as r1lite")
    ap.add_argument("--dest", default="xpolicylab", help="prefix in the output bucket")
    ap.add_argument("--bucket", default="processed")
    ap.add_argument("--limit", type=int, default=0, help="max episodes per task in the output (0 = all)")
    args = ap.parse_args()

    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    archives = sorted(f for f in os.listdir(args.raw_dir) if f.endswith(".tar.gz") and fnmatch.fnmatch(f[: -len(".tar.gz")], args.tasks))
    print(f"{len(archives)} Galaxea task archives match {args.tasks}", flush=True)
    for archive in archives:
        task = archive[: -len(".tar.gz")]
        path = os.path.join(args.raw_dir, archive)
        st = os.stat(path)
        source = {"archive": archive, "size": st.st_size, "mtime": int(st.st_mtime)}
        prefix = f"{args.dest}/{args.bench}/{task}/{args.env_cfg}"
        mkey = f"{prefix}/manifest.json"
        man = load_manifest(s3, args.bucket, mkey) or {"source": None, "episodes": {}, "complete": False}
        if man["source"] == source and (man["complete"] or (args.limit and len(man["episodes"]) >= args.limit)):
            print(f"{task}: unchanged, {len(man['episodes'])} episodes already converted", flush=True)
            continue
        work = tempfile.mkdtemp(prefix=f"galaxea-{task}-", dir=args.scratch)
        try:
            with tarfile.open(path) as tar:
                tar.extractall(work, filter="data")
            subset_dir = os.path.join(work, task) if os.path.isdir(os.path.join(work, task)) else work
            info = json.load(open(f"{subset_dir}/meta/info.json"))
            task_of = {t["task_index"]: t["task"] for t in map(json.loads, open(f"{subset_dir}/meta/tasks.jsonl"))}
            episodes = sorted((json.loads(l) for l in open(f"{subset_dir}/meta/episodes.jsonl") if l.strip()), key=lambda e: e["episode_index"])
            todo = [e for e in episodes if str(e["episode_index"]) not in man["episodes"]]
            room = (args.limit - len(man["episodes"])) if args.limit else len(todo)
            todo = todo[: max(room, 0)]
            frames, nxt = 0, (max(man["episodes"].values()) + 1) if man["episodes"] else 0
            for ep in todo:
                key = f"{prefix}/data/episode_{nxt:07d}.hdf5"
                with tempfile.NamedTemporaryFile(suffix=".hdf5", dir=args.scratch) as tmp:
                    frames += convert_episode(subset_dir, info, ep, task_of, tmp.name)
                    s3.upload_file(tmp.name, args.bucket, key)
                man["episodes"][str(ep["episode_index"])] = nxt
                man["source"], nxt = source, nxt + 1
                s3.put_object(Bucket=args.bucket, Key=mkey, Body=json.dumps(man, indent=1).encode())
            man["source"] = source
            man["complete"] = len(man["episodes"]) >= len(episodes)
            s3.put_object(Bucket=args.bucket, Key=mkey, Body=json.dumps(man, indent=1).encode())
            print(f"{task}: +{len(todo)} episodes ({frames} frames), {len(man['episodes'])}/{len(episodes)} in s3://{args.bucket}/{prefix}/data", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
