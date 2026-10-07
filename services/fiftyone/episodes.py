"""The canonical episode layout every viewer reads, and helpers to write it.

Raw datasets come in any format; converters turn each episode into

    processed/episodes/<dataset>/<...>/<episode_id>/
        <camera>.mp4        one per camera, frames on a common clock
        episode.json        dataset, episode_id, source, task, tasks, fps, frames,
                            duration_s, cameras{name: {frames, fps, width, height}}, extra
        signals.parquet     long format: t (s), group (e.g. "state.left_arm"), index, value

and FiftyOne / Rerun are built from that alone (ingest_episodes.py, episode_rrd.py).
Writers upload through the S3 gateway so the objects get ETags and events.
"""

import io
import json
import os
import tempfile

import boto3
import pyarrow as pa
import pyarrow.parquet as pq

BUCKET = "processed"
PREFIX = "episodes"


def s3():
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])


def key(dataset, episode_id, name):
    return f"{PREFIX}/{dataset}/{episode_id}/{name}"


def exists(client, dataset, episode_id):
    try:
        client.head_object(Bucket=BUCKET, Key=key(dataset, episode_id, "episode.json"))
        return True
    except client.exceptions.ClientError:
        return False


def write_signals(client, dataset, episode_id, rows):
    """rows: iterable of (t, group, index, value)."""
    t, g, i, v = zip(*rows) if rows else ((), (), (), ())
    table = pa.table({"t": pa.array(t, pa.float64()), "group": pa.array(g, pa.string()),
                      "index": pa.array(i, pa.int32()), "value": pa.array(v, pa.float64())})
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    buf.seek(0)
    client.upload_fileobj(buf, BUCKET, key(dataset, episode_id, "signals.parquet"))
    return len(t)


def write_json(client, dataset, episode_id, info):
    client.put_object(Bucket=BUCKET, Key=key(dataset, episode_id, "episode.json"),
                      Body=json.dumps(info, indent=1, ensure_ascii=False).encode())


def upload_file(client, dataset, episode_id, name, path):
    client.upload_file(path, BUCKET, key(dataset, episode_id, name))


def flatten(prefix, value):
    """Yield (group, index, scalar) for a numeric value or (nested) array."""
    try:
        seq = list(value)
    except TypeError:
        yield prefix, 0, float(value)
        return
    for i, v in enumerate(seq):
        try:
            list(v)
            for g, j, s in flatten(f"{prefix}.{i}", v):
                yield g, j, s
        except TypeError:
            yield prefix, i, float(v)


def tmpdir():
    return tempfile.TemporaryDirectory()
