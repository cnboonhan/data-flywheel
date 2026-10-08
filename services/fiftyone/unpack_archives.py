#!/usr/bin/env python
"""Unpack tar archives from the raw bucket into the processed bucket.

Reads each archive from the read-only /buckets mount (fast, local) and
streams its members to the S3 gateway, so the processed objects get proper
ETags and bucket events. An archive is skipped when its marker object
`<dest>/<archive name>/.unpacked` already exists.

Example: Galaxea ships one LeRobot v2 dataset per task as
raw/open_datasets/galaxea-open-world-r1lite/lerobot/<task>.tar.gz; this writes
processed/galaxea-open-world-r1lite/<task>/...
"""

import argparse
import glob
import os
import shutil
import tarfile
import tempfile

import boto3


def unpack(s3, bucket, dest, archive):
    name = os.path.basename(archive)
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    marker = f"{dest}/{name}/.unpacked"
    try:
        s3.head_object(Bucket=bucket, Key=marker)
        print(f"skip {name}: already unpacked")
        return 0
    except s3.exceptions.ClientError:
        pass

    n = 0
    with tarfile.open(archive, "r|*") as tar:  # streaming mode: no seeking back through the gzip
        for m in tar:
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name  # drop the archive's top dir
            # boto3 needs a seekable body (multipart, checksums); the tar stream
            # isn't, so spool each member through a temp file.
            with tempfile.TemporaryFile() as tmp:
                shutil.copyfileobj(tar.extractfile(m), tmp)
                tmp.seek(0)
                s3.upload_fileobj(tmp, bucket, f"{dest}/{name}/{rel}")
            n += 1
    s3.put_object(Bucket=bucket, Key=marker, Body=b"")
    print(f"{name}: {n} files")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archives", required=True, help="glob of archives under /buckets/raw, e.g. 'open_datasets/galaxea-open-world-r1lite/lerobot/*.tar.gz'")
    ap.add_argument("--dest", required=True, help="prefix in the processed bucket, e.g. galaxea-open-world-r1lite")
    ap.add_argument("--bucket", default="processed")
    ap.add_argument("--root", default="/buckets/raw")
    args = ap.parse_args()

    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    archives = sorted(glob.glob(f"{args.root}/{args.archives}"))
    print(f"{len(archives)} archives match {args.archives}")
    total = sum(unpack(s3, args.bucket, args.dest.strip("/"), a) for a in archives)
    print(f"uploaded {total} files to s3://{args.bucket}/{args.dest}/")


if __name__ == "__main__":
    main()
