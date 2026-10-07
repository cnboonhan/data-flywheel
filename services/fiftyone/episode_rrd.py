#!/usr/bin/env python
"""Build a Rerun recording (.rrd) for every canonical episode that lacks one.

For each processed/episodes/<dataset>/.../episode.json, logs the camera mp4s
as video assets and every signal group from signals.parquet as scalar series
on a shared "time" timeline, and uploads
processed/rerun/<dataset>/<episode_id>.rrd. The Rerun service serves those at
https://<host>:8446/data/<dataset>/<episode_id>.rrd, which the viewer opens
with ?url=<that>.

Needs: pip install rerun-sdk pyarrow boto3
"""

import argparse
import glob
import json
import os
import tempfile

import boto3
import pyarrow.parquet as pq
import rerun as rr

BUCKET, PREFIX = "processed", "rerun"


def build(episode_dir, out):
    info = json.load(open(f"{episode_dir}/episode.json"))
    rr.init(f"{info['dataset']}/{info['episode_id']}", spawn=False)
    rr.save(out)
    rr.log("episode", rr.TextDocument(json.dumps({k: v for k, v in info.items() if k != "cameras"}, indent=1, ensure_ascii=False)), static=True)
    for cam, c in info.get("cameras", {}).items():
        path = f"{episode_dir}/{cam}.mp4"
        if not os.path.exists(path):
            continue
        asset = rr.AssetVideo(path=path)
        rr.log(f"video/{cam}", asset, static=True)
        stamps = asset.read_frame_timestamps_nanos()
        rr.send_columns(f"video/{cam}", indexes=[rr.TimeColumn("time", duration=1e-9 * stamps)],
                        columns=rr.VideoFrameReference.columns_nanos(stamps))
    sig = f"{episode_dir}/signals.parquet"
    if os.path.exists(sig):
        t = pq.read_table(sig).to_pandas()
        for (group, index), g in t.groupby(["group", "index"], sort=False):
            g = g.sort_values("t")
            rr.send_columns(f"signals/{group}/{index}", indexes=[rr.TimeColumn("time", duration=g["t"].to_numpy())],
                            columns=rr.Scalars.columns(scalars=g["value"].to_numpy()))
    rr.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default="*/**/episode.json", help="glob under processed/episodes")
    ap.add_argument("--root", default="/buckets/processed/episodes")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    metas = sorted(glob.glob(f"{args.root}/{args.path}", recursive=True))
    print(f"{len(metas)} episodes under {args.root}/{args.path}")
    done = 0
    with tempfile.TemporaryDirectory() as tmp:
        for meta in metas:
            d = os.path.dirname(meta)
            rel = os.path.relpath(d, args.root)
            key = f"{PREFIX}/{rel}.rrd"
            if not args.force:
                try:
                    s3.head_object(Bucket=BUCKET, Key=key)
                    continue
                except s3.exceptions.ClientError:
                    pass
            out = f"{tmp}/episode.rrd"
            build(d, out)
            s3.upload_file(out, BUCKET, key)
            done += 1
    print(f"{done} recordings written to s3://{BUCKET}/{PREFIX}/")


if __name__ == "__main__":
    main()
