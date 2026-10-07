#!/usr/bin/env python
"""Ingest a LeRobot v3 dataset from the raw bucket into FiftyOne.

LeRobot v3 packs many episodes into one mp4 per camera, so a FiftyOne sample
is one such mp4 (fields: chunk, part, camera) and each episode becomes a
TemporalDetection on it (label = task, support = frame range, episode_index,
length). The saved view "episodes" (to_clips) shows one clip per episode per
camera. Idempotent: videos already in the dataset are skipped.

Runs inside an Actions job (see gitea/examples/ingest-lerobot.yml) with the
buckets mounted at /buckets and FIFTYONE_DATABASE_URI pointing at the stack's
Mongo, but works anywhere those two hold.
"""

import argparse
import glob
import json
import os

import fiftyone as fo
import pandas as pd


def episode_detections(eps, cam, fps, task_of):
    dets = []
    for _, e in eps.iterrows():
        task = e["tasks"][0]
        first = int(round(e[f"videos/{cam}/from_timestamp"] * fps)) + 1  # FiftyOne frames are 1-based
        last = max(int(round(e[f"videos/{cam}/to_timestamp"] * fps)), first)
        dets.append(
            fo.TemporalDetection(
                label=task_of.get(task, str(task)),
                support=[first, last],
                episode_index=int(e["episode_index"]),
                length=int(e["length"]),
            )
        )
    return dets


def ingest_part(ds, part, have):
    info = json.load(open(f"{part}/meta/info.json"))
    fps = info["fps"]
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    tasks = pd.read_parquet(f"{part}/meta/tasks.parquet")
    # tasks.parquet is indexed by task text with a task_index column; episodes
    # reference tasks by text in this dataset, by index in others.
    task_of = {t: t for t in tasks.index}
    task_of.update({int(i): t for t, i in tasks["task_index"].items()})
    eps = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{part}/meta/episodes/*/*.parquet")))
    chunk, pname = part.rstrip("/").split("/")[-2:]

    samples = []
    for cam in cams:
        groups = eps.groupby([f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"])
        for (c, f), g in groups:
            path = f"{part}/" + info["video_path"].format(video_key=cam, chunk_index=int(c), file_index=int(f))
            if path in have or not os.path.exists(path):
                continue
            samples.append(
                fo.Sample(
                    filepath=path,
                    chunk=chunk,
                    part=pname,
                    camera=cam.split(".")[-1],
                    episodes=fo.TemporalDetections(detections=episode_detections(g, cam, fps, task_of)),
                )
            )
    if samples:
        ds.add_samples(samples)
        print(f"{chunk}/{pname}: added {len(samples)} videos, {len(eps)} episodes each")
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="directory name under --root, e.g. HiFi-UMI-2K")
    ap.add_argument("--chunks", default="*", help="glob of chunk directories to ingest (default: all)")
    ap.add_argument("--root", default="/buckets/raw", help="bucket directory holding the dataset")
    ap.add_argument("--name", help="FiftyOne dataset name (default: --dataset)")
    args = ap.parse_args()
    name = args.name or args.dataset

    ds = fo.load_dataset(name) if fo.dataset_exists(name) else fo.Dataset(name, persistent=True)
    have = set(ds.values("filepath"))
    parts = sorted(glob.glob(f"{args.root}/{args.dataset}/{args.chunks}/part-*"))
    print(f"{len(parts)} parts in {args.dataset}/{args.chunks}, {len(have)} videos already ingested")

    added = [s for part in parts for s in ingest_part(ds, part, have)]
    if added:
        ds.select(s.id for s in added).compute_metadata()
    # A saved clips view is materialised when saved and doesn't pick up new
    # samples, so rebuild it whenever something was added.
    if added or "episodes" not in ds.list_saved_views():
        if "episodes" in ds.list_saved_views():
            ds.delete_saved_view("episodes")
        ds.save_view("episodes", ds.to_clips("episodes"))
    print(ds)
    print("episode clips:", ds.load_saved_view("episodes").count())


if __name__ == "__main__":
    main()
