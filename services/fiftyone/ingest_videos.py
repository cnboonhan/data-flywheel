#!/usr/bin/env python
"""Ingest plain video files into FiftyOne, one sample per mp4.

--path is a glob under --root matching mp4 files. If the video's directory
holds an episode.json (as written by convert_mcap.py), its top-level string
and number fields become sample fields, and the file stem becomes `camera`.
Idempotent: videos already in the dataset are skipped.
"""

import argparse
import glob
import json
import os

import fiftyone as fo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="FiftyOne dataset name, e.g. h2rc")
    ap.add_argument("--path", required=True, help="glob under --root, e.g. 'h2rc/**/*.mp4'")
    ap.add_argument("--root", default="/buckets/processed")
    args = ap.parse_args()

    ds = fo.load_dataset(args.name) if fo.dataset_exists(args.name) else fo.Dataset(args.name, persistent=True)
    have = set(ds.values("filepath"))
    videos = sorted(glob.glob(f"{args.root}/{args.path}", recursive=True))
    print(f"{len(videos)} videos under {args.root}/{args.path}, {len(have)} already in {args.name}")

    samples, infos = [], {}
    for path in videos:
        if path in have:
            continue
        d = os.path.dirname(path)
        if d not in infos:
            f = f"{d}/episode.json"
            infos[d] = json.load(open(f)) if os.path.exists(f) else {}
        info = infos[d]
        fields = {k: v for k, v in info.items() if isinstance(v, (str, int, float)) and k != "cameras"}
        cam = os.path.splitext(os.path.basename(path))[0]
        # `frames` is reserved on video samples (per-frame labels), so rename.
        fields.update({("frame_count" if k == "frames" else k): v for k, v in info.get("cameras", {}).get(cam, {}).items()})
        samples.append(fo.Sample(filepath=path, dataset=args.name, camera=cam, **fields))
    if samples:
        ds.add_samples(samples)
        ds.select(s.id for s in samples).compute_metadata()
        print(f"added {len(samples)} videos")
    print(ds)


if __name__ == "__main__":
    main()
