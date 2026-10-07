#!/usr/bin/env python
"""Ingest canonical episodes (processed/episodes/<dataset>/...) into FiftyOne.

One FiftyOne dataset per --dataset, grouped: a group per episode with one
slice per camera, so the App shows an episode once and switches cameras.
Fields come from episode.json (task, tasks, robot, fps, frames, duration_s,
format, source, day, ...) plus a few signal summaries and, when a Rerun
recording exists, `rerun_url` pointing at the Rerun service.
Idempotent: episodes already in the dataset are skipped.
"""

import argparse
import glob
import json
import os

import fiftyone as fo
import pyarrow.parquet as pq


def summaries(path):
    if not os.path.exists(path):
        return {}
    t = pq.read_table(path).to_pandas()
    out = {"signal_groups": int(t["group"].nunique())}
    for g in ("action.left_gripper", "action.right_gripper", "observation.state.left_gripper", "observation.state.right_gripper"):
        s = t[t["group"] == g]["value"]
        if len(s):
            out[g.replace("observation.", "").replace(".", "_") + "_range"] = float(s.max() - s.min())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="episode dataset name, e.g. galaxea-open-world-r1lite")
    ap.add_argument("--name", help="FiftyOne dataset name (default: episodes/<dataset>)")
    ap.add_argument("--root", default="/buckets/processed")
    ap.add_argument("--rerun-base", default=os.environ.get("RERUN_BASE", "https://localhost:8446"), help="Rerun service URL as seen from the browser")
    args = ap.parse_args()
    name = args.name or f"episodes/{args.dataset}"

    if fo.dataset_exists(name):
        ds = fo.load_dataset(name)
    else:
        ds = fo.Dataset(name, persistent=True)
        ds.add_group_field("camera")
    have = set(ds.values("episode_id")) if ds.has_field("episode_id") else set()
    metas = sorted(glob.glob(f"{args.root}/episodes/{args.dataset}/**/episode.json", recursive=True))
    print(f"{len(metas)} episodes on disk, {len(have)} already in {name}")

    samples = []
    for meta in metas:
        info = json.load(open(meta))
        eid = info["episode_id"]
        if eid in have:
            continue
        d = os.path.dirname(meta)
        # `frames` is reserved on video samples (per-frame labels), hence frame_count.
        fields = {("frame_count" if k == "frames" else k): v for k, v in info.items()
                  if isinstance(v, (str, int, float, list)) and k not in ("cameras", "bags")}
        fields.update(summaries(f"{d}/signals.parquet"))
        rrd = f"{args.root}/rerun/{args.dataset}/{eid}.rrd"
        if os.path.exists(rrd):
            fields["rerun_url"] = f"{args.rerun_base}/?url={args.rerun_base}/data/{args.dataset}/{eid}.rrd"
        group = fo.Group()
        for cam, c in info.get("cameras", {}).items():
            path = f"{d}/{cam}.mp4"
            if os.path.exists(path):
                samples.append(fo.Sample(filepath=path, camera=group.element(cam), camera_name=cam, cam_fps=c.get("fps"), cam_frames=c.get("frames"), **fields))
    if samples:
        ds.add_samples(samples)
        ds.select(s.id for s in samples).compute_metadata()
        print(f"added {len(samples)} videos")
    print(ds)


if __name__ == "__main__":
    main()
