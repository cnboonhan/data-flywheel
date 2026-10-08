#!/usr/bin/env python
"""Convert XPolicyLab xspark v1.0 HDF5 episodes (RoboDojo sim data, or anything
convert_*_xspark.py produced) into the canonical episode layout.

--path is a glob under --root matching directories that hold episode_*.hdf5
(e.g. 'open_datasets/robodojo/stack_bowls/arx_x5/data'). Each file becomes
processed/episodes/<dataset>/<dir name>/episode_<n>/ with one mp4 per
vision/<cam>/colors (JPEG frames piped through ffmpeg), signals from every
state/ and action/ array, and the instruction as task.

Needs: pip install h5py numpy imageio-ffmpeg boto3 pyarrow
"""

import argparse
import glob
import json
import os
import subprocess

import h5py
import imageio_ffmpeg
import numpy as np

import episodes as ep


def jpeg_frames(ds):
    """Yield raw JPEG bytes from a colors dataset (fixed-width S, vlen uint8 or padded uint8 rows)."""
    for row in ds:
        b = row.tobytes() if isinstance(row, np.ndarray) else bytes(row)
        yield b.rstrip(b"\0")


def write_video(ds, fps, out):
    ffmpeg = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-r", str(fps), "-i", "-",
                               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    n = 0
    for b in jpeg_frames(ds):
        ffmpeg.stdin.write(b)
        n += 1
    ffmpeg.stdin.close()
    ffmpeg.wait()
    return n


def scalar(ds):
    v = ds[()]
    v = v.decode() if isinstance(v, bytes) else str(v)
    try:
        return json.loads(v)
    except (ValueError, TypeError):
        return v


def convert(client, dataset, name, path, idx, tmp):
    eid = f"{name}/episode_{idx:06d}"
    if ep.exists(client, dataset, eid):
        return False
    with h5py.File(path) as f:
        fps = int(f["additional_info/frequency"][()]) if "additional_info/frequency" in f else 25
        instr = scalar(f["instructions"]) if "instructions" in f else (scalar(f["instruction"]) if "instruction" in f else [])
        tasks = instr if isinstance(instr, list) else [instr]
        cams, frames = {}, 0
        for cam in f["vision"]:
            if "colors" not in f["vision"][cam]:
                continue
            out = f"{tmp}/{cam}.mp4"
            n = write_video(f["vision"][cam]["colors"], fps, out)
            ep.upload_file(client, dataset, eid, f"{cam}.mp4", out)
            shape = [int(x) for x in f["vision"][cam]["shape"][()]] if "shape" in f["vision"][cam] else None   # numpy ints aren't JSON
            cams[cam] = {"frames": n, "fps": fps, "height": shape[0] if shape else None, "width": shape[1] if shape else None}
            frames = max(frames, n)
        rows = []
        for group in ("state", "action"):
            if group not in f:
                continue
            for key in f[group]:
                arr = np.asarray(f[group][key][()], dtype=float)
                if arr.ndim == 1:
                    arr = arr[:, None]
                t = np.arange(len(arr)) / fps
                for i in range(arr.shape[1]):
                    rows += [(float(tt), f"{group}.{key}", i, float(v)) for tt, v in zip(t, arr[:, i])]
        n = ep.write_signals(client, dataset, eid, rows)
        ep.write_json(client, dataset, eid, {
            "dataset": dataset, "episode_id": eid, "source": os.path.relpath(path, "/buckets"), "format": "xspark-v1.0",
            "robot": name.split("/")[-2] if "/" in name else None, "task": tasks[0] if tasks else None, "tasks": tasks,
            "fps": fps, "frames": frames, "duration_s": round(frames / fps, 3), "cameras": cams, "signal_rows": n})
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="name under processed/episodes/, e.g. robodojo")
    ap.add_argument("--path", required=True, help="glob under --root of directories holding episode_*.hdf5, e.g. 'open_datasets/robodojo/stack_bowls/arx_x5/data'")
    ap.add_argument("--root", default="/buckets/raw")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = ep.s3()
    dirs = sorted(d for d in glob.glob(f"{args.root}/{args.path}") if glob.glob(f"{d}/episode_*.hdf5"))
    print(f"{len(dirs)} episode directories under {args.root}/{args.path}")
    with ep.tmpdir() as tmp:
        for d in dirs:
            files = sorted(glob.glob(f"{d}/episode_*.hdf5"))[: args.limit or None]
            # open_datasets/robodojo/stack_bowls/arx_x5/data -> stack_bowls/arx_x5
            name = "/".join(os.path.relpath(d, ep.dataset_dir(d, args.root, args.dataset)).split("/")[:-1]) or os.path.basename(d)
            done = sum(convert(client, args.dataset, name, p, i, tmp) for i, p in enumerate(files))
            print(f"{name}: {done} new of {len(files)} episodes -> s3://processed/episodes/{args.dataset}/{name}/")


if __name__ == "__main__":
    main()
