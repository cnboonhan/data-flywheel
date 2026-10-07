#!/usr/bin/env python
"""Ingest LeRobot datasets (v2.x and v3) from the buckets into FiftyOne.

--path is a glob under --root; every directory it matches that holds
meta/info.json is a LeRobot dataset root (for v3, that's each chunk/part).

v3 packs many episodes into one mp4 per camera, so a sample is one such mp4
(fields: dataset, chunk, part, camera) and each episode is a TemporalDetection
on it (label = task, support = frame range, episode_index, length). The saved
view "episodes" (to_clips) shows one clip per episode per camera.

v2 has one mp4 per episode per camera, so a sample is an episode video
(fields: dataset, episode_index, camera, task, tasks, length).

Idempotent: videos already in the FiftyOne dataset are skipped. Runs inside an
Actions job (see gitea/examples/ingest-lerobot.yml) with the buckets mounted
at /buckets and FIFTYONE_DATABASE_URI pointing at the stack's Mongo, but works
anywhere those two hold.
"""

import argparse
import glob
import json
import os

import fiftyone as fo
import pandas as pd


def video_keys(info):
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- v3: one mp4 per camera holding many episodes -----------------------------

def v3_detections(eps, cam, fps, task_of):
    dets = []
    for _, e in eps.iterrows():
        task = e["tasks"][0]
        first = int(round(e[f"videos/{cam}/from_timestamp"] * fps)) + 1  # FiftyOne frames are 1-based
        last = max(int(round(e[f"videos/{cam}/to_timestamp"] * fps)), first)
        dets.append(fo.TemporalDetection(label=task_of.get(task, str(task)), support=[first, last],
                                         episode_index=int(e["episode_index"]), length=int(e["length"])))
    return dets


def v3_samples(root, info, have, dataset):
    tasks = pd.read_parquet(f"{root}/meta/tasks.parquet")
    # tasks.parquet is indexed by task text with a task_index column; episodes
    # reference tasks by text in some datasets and by index in others.
    task_of = {t: t for t in tasks.index}
    task_of.update({int(i): t for t, i in tasks["task_index"].items()})
    eps = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")))
    chunk, part = root.rstrip("/").split("/")[-2:]
    samples = []
    for cam in video_keys(info):
        for (c, f), g in eps.groupby([f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"]):
            path = f"{root}/" + info["video_path"].format(video_key=cam, chunk_index=int(c), file_index=int(f))
            if path in have or not os.path.exists(path):
                continue
            samples.append(fo.Sample(filepath=path, dataset=dataset, chunk=chunk, part=part, camera=cam.split(".")[-1],
                                     episodes=fo.TemporalDetections(detections=v3_detections(g, cam, info["fps"], task_of))))
    return samples, f"{len(eps)} episodes per video"


# --- v2: one mp4 per episode per camera ---------------------------------------

def v2_samples(root, info, have, dataset):
    task_of = {t["task_index"]: t["task"] for t in read_jsonl(f"{root}/meta/tasks.jsonl")}
    eps = read_jsonl(f"{root}/meta/episodes.jsonl")
    name = os.path.basename(root.rstrip("/"))
    samples = []
    for e in eps:
        idx = e["episode_index"]
        tasks = [task_of.get(t, str(t)) for t in e.get("tasks", [])]
        for cam in video_keys(info):
            path = f"{root}/" + info["video_path"].format(episode_chunk=idx // info["chunks_size"], episode_index=idx, video_key=cam)
            if path in have or not os.path.exists(path):
                continue
            samples.append(fo.Sample(filepath=path, dataset=dataset, subset=name, episode_index=idx, camera=cam.split(".")[-1],
                                     task=tasks[0] if tasks else None, tasks=tasks, length=int(e.get("length", 0))))
    return samples, f"{len(eps)} episodes"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="FiftyOne dataset name, e.g. HiFi-UMI-2K")
    ap.add_argument("--path", required=True, help="glob under --root matching LeRobot roots, e.g. 'HiFi-UMI-2K/chunk-000*/part-*' or 'galaxea-open-world-r1lite/*'")
    ap.add_argument("--root", default="/buckets/raw", help="bucket directory (default: /buckets/raw)")
    args = ap.parse_args()

    ds = fo.load_dataset(args.name) if fo.dataset_exists(args.name) else fo.Dataset(args.name, persistent=True)
    have = set(ds.values("filepath"))
    roots = sorted(os.path.dirname(os.path.dirname(p)) for p in glob.glob(f"{args.root}/{args.path}/meta/info.json"))
    print(f"{len(roots)} LeRobot roots under {args.root}/{args.path}, {len(have)} videos already in {args.name}")

    added, v3 = [], False
    for root in roots:
        info = json.load(open(f"{root}/meta/info.json"))
        is_v3 = info["codebase_version"].startswith("v3")
        v3 = v3 or is_v3
        samples, what = (v3_samples if is_v3 else v2_samples)(root, info, have, args.name)
        if samples:
            ds.add_samples(samples)
            added += samples
            print(f"{os.path.relpath(root, args.root)}: added {len(samples)} videos ({what})")
    if added:
        ds.select(s.id for s in added).compute_metadata()
    # A saved clips view is materialised when saved and doesn't pick up new
    # samples, so rebuild it whenever something was added.
    if v3 and (added or "episodes" not in ds.list_saved_views()):
        if "episodes" in ds.list_saved_views():
            ds.delete_saved_view("episodes")
        ds.save_view("episodes", ds.to_clips("episodes"))
        print("episode clips:", ds.load_saved_view("episodes").count())
    print(ds)


if __name__ == "__main__":
    main()
