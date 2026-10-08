#!/usr/bin/env python
"""Promote xspark episodes that already sit in raw (RoboDojo's own data) into
the XPolicyLab training tree, processed/xpolicylab/<bench>/<task>/<env_cfg>/data/,
by server-side S3 copy. Raw stays untouched; training reads the copy.

    python promote_xspark.py --src robodojo/stack_bowls/arx_x5/data --bench RoboDojo --task stack_bowls --env-cfg arx_x5
"""

import argparse
import os

import boto3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="prefix in the raw bucket holding episode_*.hdf5")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--env-cfg", default="arx_x5")
    ap.add_argument("--limit", type=int, default=0, help="episodes (0 = all)")
    args = ap.parse_args()

    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    src = args.src.strip("/") + "/"
    keys = sorted(o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket="raw", Prefix=src) for o in p.get("Contents", [])
                  if o["Key"].endswith(".hdf5"))
    if args.limit:
        keys = keys[: args.limit]
    dest = f"xpolicylab/{args.bench}/{args.task}/{args.env_cfg}/data"
    have = {o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket="processed", Prefix=dest + "/") for o in p.get("Contents", [])}
    n = 0
    for i, key in enumerate(keys):
        dk = f"{dest}/episode_{i:07d}.hdf5"   # XPolicyLab wants contiguous numbering from 0
        if dk in have:
            continue
        s3.copy({"Bucket": "raw", "Key": key}, "processed", dk)
        n += 1
    print(f"{len(keys)} episodes in s3://raw/{src}; {n} copied to s3://processed/{dest}/")


if __name__ == "__main__":
    main()
