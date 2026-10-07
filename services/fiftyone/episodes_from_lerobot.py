#!/usr/bin/env python
"""Convert LeRobot datasets (v2.x and v3) into the canonical episode layout.

--path is a glob under --root; every match holding meta/info.json is a LeRobot
root (for v3, each chunk/part). Output goes to
processed/episodes/<dataset>/<root name>/episode_<n>/ (see episodes.py).

v2: one mp4 per episode per camera already; uploaded as is.
v3: many episodes per mp4, so each episode is cut out with ffmpeg (re-encoded;
    the source GOP doesn't align with episode boundaries).
Signals: every non-image column of the episode's parquet rows, flattened.

Needs: pip install pyarrow pandas boto3 imageio-ffmpeg
"""

import argparse
import glob
import json
import os
import subprocess

import imageio_ffmpeg
import pandas as pd
import pyarrow.parquet as pq

import episodes as ep


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def video_keys(info):
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def cam_name(key):
    return key.split(".")[-1].replace("observation_", "")


def signal_rows(df):
    skip = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    cols = [c for c in df.columns if not c.startswith("observation.images") and c not in skip and df[c].dtype != object or
            (df[c].dtype == object and c not in skip and not c.startswith("observation.images") and hasattr(df[c].iloc[0], "__len__") and not isinstance(df[c].iloc[0], str))]
    rows = []
    ts = df["timestamp"].to_numpy(dtype=float)
    for c in cols:
        for t, v in zip(ts, df[c].to_numpy()):
            for g, i, s in ep.flatten(c, v):
                rows.append((float(t), g, i, s))
    return rows


def probe(path):
    out = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe().replace("ffmpeg", "ffprobe") if False else imageio_ffmpeg.get_ffmpeg_exe(), "-i", path],
                         capture_output=True, text=True).stderr
    w = h = None
    for line in out.splitlines():
        if "Video:" in line:
            import re
            m = re.search(r"(\d{2,5})x(\d{2,5})", line)
            if m:
                w, h = int(m.group(1)), int(m.group(2))
    return w, h


def cut(src, start, end, dst):
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)


def convert_v2(client, dataset, root, info, name, limit):
    task_of = {t["task_index"]: t["task"] for t in read_jsonl(f"{root}/meta/tasks.jsonl")}
    eps = sorted(read_jsonl(f"{root}/meta/episodes.jsonl"), key=lambda e: e["episode_index"])
    done = 0
    for e in eps[: limit or None]:
        idx = e["episode_index"]
        eid = f"{name}/episode_{idx:06d}"
        if ep.exists(client, dataset, eid):
            continue
        chunk = idx // info["chunks_size"]
        df = pd.read_parquet(f"{root}/" + info["data_path"].format(episode_chunk=chunk, episode_index=idx))
        cams = {}
        for key in video_keys(info):
            path = f"{root}/" + info["video_path"].format(episode_chunk=chunk, episode_index=idx, video_key=key)
            if not os.path.exists(path):
                continue
            cam = cam_name(key)
            ep.upload_file(client, dataset, eid, f"{cam}.mp4", path)
            w, h = probe(path)
            cams[cam] = {"frames": len(df), "fps": info["fps"], "width": w, "height": h, "source": os.path.relpath(path, root)}
        tasks = [task_of.get(t, str(t)) for t in e.get("tasks", [])]
        n = ep.write_signals(client, dataset, eid, signal_rows(df))
        ep.write_json(client, dataset, eid, {
            "dataset": dataset, "episode_id": eid, "source": os.path.relpath(root, "/buckets"), "format": "lerobot-" + info["codebase_version"],
            "robot": info.get("robot_type"), "task": tasks[0] if tasks else None, "tasks": tasks, "fps": info["fps"], "frames": len(df),
            "duration_s": round(len(df) / info["fps"], 3), "cameras": cams, "signal_rows": n})
        done += 1
    return done, len(eps)


def convert_v3(client, dataset, root, info, name, limit, tmp):
    tasks = pd.read_parquet(f"{root}/meta/tasks.parquet")
    task_of = {t: t for t in tasks.index}
    task_of.update({int(i): t for t, i in tasks["task_index"].items()})
    eps = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")))
    eps = eps.sort_values("episode_index")
    fps = info["fps"]
    data_cache = {}
    done = 0
    for _, e in eps.head(limit or len(eps)).iterrows():
        idx = int(e["episode_index"])
        eid = f"{name}/episode_{idx:06d}"
        if ep.exists(client, dataset, eid):
            continue
        dpath = f"{root}/" + info["data_path"].format(chunk_index=int(e["data/chunk_index"]), file_index=int(e["data/file_index"]))
        if dpath not in data_cache:
            data_cache.clear()
            data_cache[dpath] = pq.read_table(dpath).to_pandas()
        df = data_cache[dpath].iloc[int(e["dataset_from_index"]):int(e["dataset_to_index"])]
        cams = {}
        for key in video_keys(info):
            src = f"{root}/" + info["video_path"].format(video_key=key, chunk_index=int(e[f"videos/{key}/chunk_index"]), file_index=int(e[f"videos/{key}/file_index"]))
            if not os.path.exists(src):
                continue
            cam = cam_name(key)
            dst = f"{tmp}/{cam}.mp4"
            cut(src, float(e[f"videos/{key}/from_timestamp"]), float(e[f"videos/{key}/to_timestamp"]), dst)
            ep.upload_file(client, dataset, eid, f"{cam}.mp4", dst)
            w, h = probe(dst)
            cams[cam] = {"frames": len(df), "fps": fps, "width": w, "height": h, "source": os.path.relpath(src, root),
                         "from_s": float(e[f"videos/{key}/from_timestamp"]), "to_s": float(e[f"videos/{key}/to_timestamp"])}
        t = [task_of.get(x, str(x)) for x in e["tasks"]]
        n = ep.write_signals(client, dataset, eid, signal_rows(df))
        ep.write_json(client, dataset, eid, {
            "dataset": dataset, "episode_id": eid, "source": os.path.relpath(root, "/buckets"), "format": "lerobot-" + info["codebase_version"],
            "robot": info.get("robot_type"), "task": t[0] if t else None, "tasks": t, "fps": fps, "frames": int(e["length"]),
            "duration_s": round(int(e["length"]) / fps, 3), "cameras": cams, "signal_rows": n})
        done += 1
    return done, len(eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="name under processed/episodes/, e.g. HiFi-UMI-2K")
    ap.add_argument("--path", required=True, help="glob under --root matching LeRobot roots")
    ap.add_argument("--root", default="/buckets/raw")
    ap.add_argument("--limit", type=int, default=0, help="episodes per root (0 = all)")
    args = ap.parse_args()

    client = ep.s3()
    roots = sorted(os.path.dirname(os.path.dirname(p)) for p in glob.glob(f"{args.root}/{args.path}/meta/info.json"))
    print(f"{len(roots)} LeRobot roots under {args.root}/{args.path}")
    with ep.tmpdir() as tmp:
        for root in roots:
            info = json.load(open(f"{root}/meta/info.json"))
            # Name the episode group after the part (v3: chunk-0000/part-0000) or the dataset dir (v2).
            rel = os.path.relpath(root, f"{args.root}")
            name = rel.replace("/", "__") if info["codebase_version"].startswith("v3") else os.path.basename(root)
            if info["codebase_version"].startswith("v3"):
                done, total = convert_v3(client, args.dataset, root, info, name, args.limit, tmp)
            else:
                done, total = convert_v2(client, args.dataset, root, info, name, args.limit)
            print(f"{rel}: {done} new of {total} episodes -> s3://processed/episodes/{args.dataset}/{name}/")


if __name__ == "__main__":
    main()
