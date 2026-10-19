"""LeRobot v3 episodes as Rerun recordings, for Rerun's web viewer: it opens one file by URL, not a LeRobot folder.

    GET /lerobot/<group>/<dataset>/<root>/episode_<N>.rrd     (Caddy strips its path token first)

Reads /raw/<group>/<dataset>/<root> (read-only): the episode's row in meta/episodes, its frames in data/, and each
camera's span of the chunk video, cut to its own H.264 file. Writes the recording to /cache once and serves it from
there after: the cameras as videos, observation.state and action as plots (one per named joint), the task as text, all
on a `time` timeline from the episode's start. The cache keeps the newest recordings within CACHE_GB.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import rerun as rr

RAW = os.environ.get("RAW_ROOT", "/raw")
CACHE = os.environ.get("CACHE_DIR", "/cache")
CACHE_GB = float(os.environ.get("CACHE_GB", "20"))   # ~4 MB a 450-frame, 6-camera episode: thousands fit; untuned
PATH = re.compile(r"^/lerobot/(?P<root>[A-Za-z0-9._/+-]+)/episode_(?P<ep>\d+)\.rrd$")
_locks, _locks_guard = {}, threading.Lock()


def episode_row(root, ep):
    """The episode's row of meta/episodes/*/*.parquet, as a dict."""
    d = pads.dataset(sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")), format="parquet")
    t = d.to_table(filter=pc.field("episode_index") == ep)
    if t.num_rows == 0:
        raise FileNotFoundError(f"episode {ep} not in {root}")
    return {k: v[0] for k, v in t.to_pydict().items()}


def cut(src, start, end, dst):
    """The [start, end) seconds of a chunk video as their own file, re-encoded so the first frame is a keyframe."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", src, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "50", dst],
                   check=True)   # crf 23, keyframe every 2 s at 25 fps: viewer-grade, small; untuned


def build(root, ep, out):
    info = json.load(open(f"{root}/meta/info.json"))
    row = episode_row(root, ep)
    data = pq.read_table(f"{root}/" + info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
    data = data.filter(pc.equal(data["episode_index"], ep))
    t = np.asarray(data["timestamp"].to_numpy(), dtype=np.float64)
    t = t - t[0]
    rec = rr.RecordingStream("lerobot", recording_id=f"{os.path.relpath(root, RAW)}/episode_{ep}")
    rec.save(out)
    task = row.get("tasks")
    rec.log("task", rr.TextDocument("\n".join(task) if isinstance(task, list) else str(task)), static=True)
    time = rr.TimeColumn("time", duration=t)
    for key in ("observation.state", "action"):
        if key not in data.column_names:
            continue
        vals = np.stack(data[key].to_numpy(zero_copy_only=False)).astype(np.float64)
        names = (info["features"][key].get("names") or {})
        names = names.get("motors", names) if isinstance(names, dict) else names
        names = names if isinstance(names, list) and len(names) == vals.shape[1] else [str(i) for i in range(vals.shape[1])]
        for i, n in enumerate(names):
            rec.send_columns(f"{key}/{n}", indexes=[time], columns=rr.Scalars.columns(scalars=vals[:, i]))
    cams = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp, ThreadPoolExecutor(len(cams) or 1) as pool:
        jobs = {}
        for k in cams:
            src = f"{root}/" + info["video_path"].format(video_key=k, chunk_index=row[f"videos/{k}/chunk_index"],
                                                        file_index=row[f"videos/{k}/file_index"])
            jobs[k] = pool.submit(cut, src, row[f"videos/{k}/from_timestamp"], row[f"videos/{k}/to_timestamp"], f"{tmp}/{k}.mp4")
        for k, job in jobs.items():
            job.result()
            asset = rr.AssetVideo(path=f"{tmp}/{k}.mp4")
            rec.log(f"cameras/{k}", asset, static=True)
            ts = asset.read_frame_timestamps_nanos()
            rec.send_columns(f"cameras/{k}", indexes=[rr.TimeColumn("time", duration=ts / 1e9)],
                             columns=rr.VideoFrameReference.columns_nanos(ts))
    rec.flush()
    rec.disconnect()


def trim_cache():
    files = sorted(glob.glob(f"{CACHE}/**/*.rrd", recursive=True), key=os.path.getmtime, reverse=True)
    total = 0
    for f in files:
        total += os.path.getsize(f)
        if total > CACHE_GB * 2**30:
            os.remove(f)


def recording(rel, ep):
    root = os.path.realpath(os.path.join(RAW, rel))
    if not root.startswith(os.path.realpath(RAW) + "/") or not os.path.exists(f"{root}/meta/info.json"):
        raise FileNotFoundError(rel)
    out = os.path.join(CACHE, rel, f"episode_{ep}.rrd")
    with _locks_guard:
        lock = _locks.setdefault(out, threading.Lock())
    with lock:   # one build per episode; later requests wait for it and read the cache
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            part = out + ".part"
            build(root, ep, part)
            os.replace(part, out)
            trim_cache()
        os.utime(out)   # newest first in trim_cache
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, b"ok", "text/plain")
        m = PATH.match(self.path.split("?")[0])
        if not m:
            return self.reply(404, b"not a LeRobot episode path", "text/plain")
        try:
            path = recording(m["root"], int(m["ep"]))
        except FileNotFoundError as e:
            return self.reply(404, str(e).encode(), "text/plain")
        except Exception as e:   # noqa: BLE001  report, keep serving
            self.log_error("build failed for %s: %r", self.path, e)
            return self.reply(500, f"could not build the recording: {e}".encode(), "text/plain")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(os.path.getsize(path)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def reply(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler).serve_forever()
