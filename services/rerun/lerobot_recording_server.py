"""LeRobot episodes (v3, or v2.x: one parquet and one video per camera per episode) as Rerun recordings, for Rerun's
web viewer: it opens one file by URL, not a LeRobot folder.

    GET /lerobot/<group>/<dataset>/<root>/episode_<N>.rrd             a dataset in the raw bucket
    GET /lerobot-processed/<path>/<root>/episode_<N>.rrd              one in the processed bucket
    (Caddy strips its path token first)
    GET /bag?bucket=<raw|processed>&path=<bag.mcap>                   a big ROS 2 bag: a page that converts it (mcap_recording.py)
    GET /bag/status?bucket=...&path=...                               and shows progress, then opens the viewer on
    GET /bag-rrd/<raw|processed>/<path to bag.mcap>.rrd               the recording (Caddy strips its path token first)
    GET /dataset/<raw|processed>/<path to root>                       a dataset's page: summary and episodes, each
    linked into the viewer (Browse Data's /data/dataset/..., behind SSO: services/browse). Caddy passes the path token
    (X-Rerun-Token), the viewer's host (X-Viewer-Host) and Browse Data's path (X-Browse-Base) for the links.

Reads /raw/<group>/<dataset>/<root> or /processed/<path>/<root> (read-only): the episode's row in meta/episodes (meta/episodes.jsonl in v2.x),
its frames in data/, and each camera's span of the chunk video (v2.x: the episode's own video), as its own H.264 file.
Writes the recording to /cache once and serves it from there after: the cameras as videos, every observation.state*, state.*
and action* feature as plots (one per named joint), the task as text, all on a `time` timeline from the episode's
start. The cache keeps the newest recordings within CACHE_GB.
"""
import glob
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import rerun as rr

import mcap_recording

RAW = os.environ.get("RAW_ROOT", "/raw")
PROCESSED = os.environ.get("PROCESSED_ROOT", "/processed")
CACHE = os.environ.get("CACHE_DIR", "/cache")
CACHE_GB = float(os.environ.get("CACHE_GB", "20"))   # ~4 MB a 450-frame, 6-camera episode: thousands fit; untuned
PATH = re.compile(r"^/(?P<kind>lerobot|lerobot-processed)/(?P<root>[A-Za-z0-9._/+-]+)/episode_(?P<ep>\d+)\.rrd$")
# per URL prefix: the bucket's mount, and its recordings' folder in the cache (raw's at the top, as before; PATH never
# matches '@', so processed's can't collide with a raw dataset)
BUCKETS = {"lerobot": (RAW, ""), "lerobot-processed": (PROCESSED, "@processed")}
DATASET = re.compile(r"^/dataset/(?P<bucket>raw|processed)/(?P<root>[A-Za-z0-9._/+-]+?)/?$")
KIND = {"raw": "lerobot", "processed": "lerobot-processed"}   # the bucket's episode URL prefix (PATH)
BAG_RRD = re.compile(r"^/bag-rrd/(?P<bucket>raw|processed)/(?P<path>.+\.mcap)\.rrd$")
BAG_BUILDS = threading.Semaphore(2)   # bags converted at once: each runs four ffmpeg encoders and a reader; untuned
BAG_ROOT = {"raw": RAW, "processed": PROCESSED}
_bags, _bags_guard = {}, threading.Lock()   # cache path -> job: state, read, total, started, error
PAGE_ROWS = 2000   # episodes per dataset page: a 2000-row table still renders and filters at once; untuned
_locks, _locks_guard = {}, threading.Lock()


def episode_row_v2(root, ep):
    """The episode's record in a v2.x root's meta/episodes.jsonl."""
    with open(f"{root}/meta/episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            if int(row["episode_index"]) == ep:
                return row
    raise FileNotFoundError(f"episode {ep} not in {root}")


def episode_row(root, ep):
    """The episode's row of meta/episodes/*/*.parquet, as a dict."""
    d = pads.dataset(sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")), format="parquet")
    t = d.to_table(filter=pc.field("episode_index") == ep)
    if t.num_rows == 0:
        raise FileNotFoundError(f"episode {ep} not in {root}")
    return {k: v[0] for k, v in t.to_pydict().items()}


def cut(src, start, end, dst):
    """The [start, end) seconds of a chunk video (end None: to its end) as their own file, re-encoded so the first frame is
    a keyframe."""
    span = ["-ss", f"{start:.3f}"] + (["-to", f"{end:.3f}"] if end is not None else [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *span, "-i", src, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", "50", dst],
                   check=True)   # crf 23, keyframe every 2 s at 25 fps: viewer-grade, small; untuned


def build(root, ep, out, rec_id):
    info = json.load(open(f"{root}/meta/info.json"))
    v2 = str(info.get("codebase_version", "")).lstrip("v").startswith("2")
    chunk = ep // int(info.get("chunks_size", 1000))
    if v2:
        row = episode_row_v2(root, ep)
        data = pq.read_table(f"{root}/" + info["data_path"].format(episode_chunk=chunk, episode_index=ep))
    else:
        row = episode_row(root, ep)
        data = pq.read_table(f"{root}/" + info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
    data = data.filter(pc.equal(data["episode_index"], ep))
    t = np.asarray(data["timestamp"].to_numpy(), dtype=np.float64)
    t = t - t[0]
    rec = rr.RecordingStream("lerobot", recording_id=rec_id)
    rec.save(out)
    task = row.get("tasks")
    rec.log("task", rr.TextDocument("\n".join(task) if isinstance(task, list) else str(task)), static=True)
    time = rr.TimeColumn("time", duration=t)
    # Galaxea's v2.x collections split them into state.* (state.left_arm, state.leader_right_ee_pose, ...)
    keys = [k for k, f in info["features"].items()
            if k.startswith(("observation.state", "action", "state."))
            and f.get("dtype") in ("float32", "float64") and k in data.column_names]
    for key in keys:
        vals = np.stack(data[key].to_numpy(zero_copy_only=False)).astype(np.float64)
        vals = vals.reshape(len(vals), -1)      # a one-value feature may be stored as a plain scalar
        names = (info["features"][key].get("names") or {})
        names = names.get("motors", names) if isinstance(names, dict) else names
        names = names if isinstance(names, list) and len(names) == vals.shape[1] else [str(i) for i in range(vals.shape[1])]
        for i, n in enumerate(names):
            rec.send_columns(f"{key}/{n}", indexes=[time], columns=rr.Scalars.columns(scalars=vals[:, i]))
    cams = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    with tempfile.TemporaryDirectory(dir=CACHE) as tmp, ThreadPoolExecutor(len(cams) or 1) as pool:
        jobs = {}
        for k in cams:
            if v2:
                src = f"{root}/" + info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=ep)
                jobs[k] = pool.submit(cut, src, 0.0, None, f"{tmp}/{k}.mp4")
            else:
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


def dataset_root(kind, rel):
    """The dataset folder for a URL path, inside its bucket and with meta/info.json, or FileNotFoundError."""
    base = BUCKETS[kind][0]
    root = os.path.realpath(os.path.join(base, rel))
    if not root.startswith(os.path.realpath(base) + "/") or not os.path.exists(f"{root}/meta/info.json"):
        raise FileNotFoundError(rel)
    return root


def episodes(root, v2):
    """(index, tasks, length in frames) of every episode, from meta/ only."""
    if v2:
        with open(f"{root}/meta/episodes.jsonl") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        d = pads.dataset(sorted(glob.glob(f"{root}/meta/episodes/*/*.parquet")), format="parquet")
        rows = d.to_table(columns=[c for c in ("episode_index", "tasks", "length") if c in d.schema.names]).to_pylist()
    out = []
    for r in rows:
        tasks = r.get("tasks") or []
        out.append((int(r["episode_index"]), tasks if isinstance(tasks, list) else [str(tasks)], int(r.get("length") or 0)))
    return sorted(out)


def dataset_page(bucket, rel, host, token, page, browse_base):
    """A dataset's summary and episode table, each episode linked into the viewer."""
    kind = KIND[bucket]
    root = dataset_root(kind, rel)
    info = json.load(open(f"{root}/meta/info.json"))
    v2 = str(info.get("codebase_version", "")).lstrip("v").startswith("2")
    fps = float(info.get("fps") or 0)
    feats = info.get("features", {})
    cams = [k for k, f in feats.items() if f.get("dtype") == "video"]
    signals = [(k, f.get("shape")) for k, f in feats.items() if f.get("dtype") in ("float32", "float64")
               and k not in ("timestamp",)]
    eps = episodes(root, v2)
    pages = max(1, -(-len(eps) // PAGE_ROWS))
    page = min(max(page, 0), pages - 1)
    e = html.escape
    viewer = f"https://{host}"

    def link(ep):
        rrd = f"{viewer}/{kind}/{token}/{rel}/episode_{ep}.rrd"
        return f"{viewer}/?url={urllib.parse.quote(rrd, safe='')}"

    rows = "".join(
        f"<tr><td class=n>{i}</td><td class=t>{e(' · '.join(t))}</td><td class=n>{n}</td>"
        f"<td class=n>{(n / fps if fps else 0):.1f} s</td><td>"
        + (f"<a class=open target=_blank rel=noopener href=\"{e(link(i))}\">open in Rerun</a>" if token else "")
        + "</td></tr>"
        for i, t, n in eps[page * PAGE_ROWS:(page + 1) * PAGE_ROWS])
    nav = "" if pages == 1 else " ".join(
        f"<a href=\"?page={p}\">{p + 1}</a>" if p != page else f"<b>{p + 1}</b>" for p in range(pages))
    frames = sum(n for _, _, n in eps)
    summary = [("Version", info.get("codebase_version", "?")), ("Robot", info.get("robot_type", "?")),
               ("Frame rate", f"{fps:g} fps"), ("Episodes", f"{len(eps)}"),
               ("Frames", f"{frames} ({frames / fps / 3600:.2f} h)" if fps else str(frames)),
               ("Tasks", str(len({x for _, t, _ in eps for x in t}))),
               ("Cameras", ", ".join(c.removeprefix("observation.images.") for c in cams) or "none")]
    browse = f"{browse_base}/{bucket}/{rel}/"   # Browse Data, on the host that serves this page
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1"><title>{e(rel.split('/')[-1])}</title><style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --link:#0a64d6; --pill:#eef4ff; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161618; --fg:#f2f2f7; --muted:#98989f; --line:#2c2c2e; --link:#6aa8ff; --pill:#1f2a3d; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px; }} h1 {{ font-size:18px; margin:0 0 4px; word-break:break-all; }}
h2 {{ font-size:15px; margin:22px 0 8px; }} a {{ color:var(--link); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
p.hint, .muted {{ color:var(--muted); }} table {{ width:100%; border-collapse:collapse; }}
td, th {{ padding:5px 8px; border-top:1px solid var(--line); text-align:left; vertical-align:top; }} th {{ color:var(--muted); font-weight:500; }}
td.n {{ text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }} dl {{ display:grid; grid-template-columns:max-content 1fr; gap:2px 16px; margin:0; }}
dt {{ color:var(--muted); }} dd {{ margin:0; word-break:break-word; }} code {{ font-size:12px; }}
a.open {{ display:inline-block; padding:1px 8px; border-radius:10px; background:var(--pill); font-size:12px; white-space:nowrap; }}
input {{ width:100%; max-width:420px; padding:6px 8px; font:inherit; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--fg); }}
</style></head><body><main>
<h1>{e(rel)}</h1><p class=hint>{e(bucket)} bucket · read from <code>meta/</code> on each visit · <a href="{e(browse)}">files</a></p>
<dl>{''.join(f'<dt>{e(k)}</dt><dd>{e(v)}</dd>' for k, v in summary)}</dl>
<h2>Signals</h2><p class=muted>{', '.join(f'<code>{e(k)}</code> {e(str(s or []))}' for k, s in signals) or 'none'}</p>
<h2>Episodes</h2><p><input id=q placeholder="Filter by task" autocomplete=off></p><p>{nav}</p>
<table><thead><tr><th>#</th><th>Task</th><th>Frames</th><th>Length</th><th></th></tr></thead><tbody id=rows>{rows}</tbody></table>
<script>
document.getElementById('q').addEventListener('input', ev => {{
  const q = ev.target.value.toLowerCase();
  for (const tr of document.getElementById('rows').rows) tr.hidden = q && !tr.cells[1].textContent.toLowerCase().includes(q);
}});
</script></main></body></html>""".encode()


def bag_paths(bucket, rel):
    """(the bag inside its bucket, its recording's cache path), or FileNotFoundError."""
    base = BAG_ROOT[bucket]
    src = os.path.realpath(os.path.join(base, rel))
    if not src.startswith(os.path.realpath(base) + "/") or not src.endswith(".mcap") or not os.path.isfile(src):
        raise FileNotFoundError(rel)
    return src, os.path.join(CACHE, "@bags", bucket, rel + ".rrd")


def bag_status(bucket, rel):
    """The bag's conversion: ready if its recording is cached, else the running job's progress; starts one if none."""
    src, out = bag_paths(bucket, rel)
    with _bags_guard:
        job = _bags.get(out)
        if job is None or job["state"] == "error" and job.get("retry"):
            if os.path.exists(out):
                os.utime(out)
                return {"state": "ready"}
            job = _bags[out] = {"state": "queued", "read": 0, "total": 0, "started": time.time(), "size": os.path.getsize(src)}
            threading.Thread(target=convert_bag, args=(src, out, job), daemon=True).start()
        elif job["state"] == "error":
            job["retry"] = True   # the next poll starts it again
        return dict(job, elapsed=time.time() - job["started"])


def convert_bag(src, out, job):
    with BAG_BUILDS:
        job["state"], job["started"] = "converting", time.time()
        try:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            part = out + ".part"
            mcap_recording.build(src, part, lambda read, total: job.update(read=read, total=total))
            os.replace(part, out)
            trim_cache()
            job["state"] = "ready"
        except Exception as e:   # noqa: BLE001  shown on the page; the next visit retries
            job.update(state="error", error=f"{type(e).__name__}: {e}")


def bag_page(bucket, rel, viewer_host, token):
    """A page that converts a big bag (polling its status) and then opens the viewer on the recording."""
    e = html.escape
    q = urllib.parse.urlencode({"bucket": bucket, "path": rel})
    rrd = f"https://{viewer_host}/bag-rrd/{token}/{bucket}/{urllib.parse.quote(rel)}.rrd"
    viewer = f"https://{viewer_host}/?url={urllib.parse.quote(rrd, safe='')}"
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width, initial-scale=1">
<title>{e(rel.split('/')[-1])}</title><style>
:root {{ --bg:#fff; --fg:#1d1d1f; --muted:#6e6e73; --line:#e5e5ea; --bar:#0a64d6; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161618; --fg:#f2f2f7; --muted:#98989f; --line:#2c2c2e; --bar:#6aa8ff; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; }}
main {{ max-width:720px; margin:0 auto; padding:32px 16px; }} h1 {{ font-size:17px; margin:0 0 4px; word-break:break-all; }}
p {{ color:var(--muted); }} .track {{ height:10px; border-radius:5px; background:var(--line); overflow:hidden; margin:18px 0 8px; }}
.fill {{ height:100%; width:0; background:var(--bar); transition:width .4s; }} #msg {{ color:var(--fg); }} a {{ color:var(--bar); }}
</style></head><body><main>
<h1>{e(rel)}</h1>
<div class=track><div class=fill id=fill></div></div><div id=msg>Starting&hellip;</div>
<p><a id=open href="{e(viewer)}" hidden>open in Rerun</a></p>
<script>
const fmt = s => s < 90 ? Math.round(s) + " s" : Math.round(s / 60) + " min";
let failures = 0;
async function poll() {{
  const msg = document.getElementById("msg"), fill = document.getElementById("fill");
  let j;
  try {{   // relative to this page (/data/bag?...): /data/bag/status, which also starts the conversion
    const r = await fetch({json.dumps("bag/status?" + q)}, {{cache: "no-store"}});   // a JS string: no HTML escaping in <script>
    if (!r.ok) throw new Error("HTTP " + r.status);
    j = await r.json(); failures = 0;
  }} catch (err) {{   // 5 in a row (~15 s): say so instead of looking stuck
    if (++failures >= 5) msg.textContent = "Can't reach the converter (" + err.message + "); still retrying\u2026";
    return setTimeout(poll, 3000);
  }}
  if (j.state === "ready") {{ fill.style.width = "100%"; msg.textContent = "Ready: opening the viewer\u2026"; document.getElementById("open").hidden = false; location.href = {json.dumps(viewer)}; return; }}
  if (j.state === "error") {{ msg.textContent = "Conversion failed: " + j.error + " (reload to try again)"; return; }}
  if (j.state === "queued") {{ msg.textContent = "Waiting for another conversion to finish (" + fmt(j.elapsed) + ")"; }}
  else {{
    const f = j.total ? j.read / j.total : 0, rate = j.elapsed > 0 ? f / j.elapsed : 0;
    fill.style.width = (100 * f).toFixed(1) + "%";
    msg.textContent = (f >= 1 ? "Encoding the last frames and writing the recording" : "Converting: " + Math.floor(100 * f) + "% of " + j.total.toLocaleString() + " messages")
      + " \u00b7 " + fmt(j.elapsed) + " so far" + (f > 0.02 && f < 1 ? " \u00b7 about " + fmt((1 - f) / rate) + " left" : "");
  }}
  setTimeout(poll, 1000);
}}
poll();
</script></main></body></html>"""


def recording(kind, rel, ep):
    base, cache_dir = BUCKETS[kind]
    root = dataset_root(kind, rel)
    out =os.path.join(CACHE, cache_dir, rel, f"episode_{ep}.rrd")
    with _locks_guard:
        lock = _locks.setdefault(out, threading.Lock())
    with lock:   # one build per episode; later requests wait for it and read the cache
        if not os.path.exists(out):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            part = out + ".part"
            build(root, ep, part, f"{'processed/' if cache_dir else ''}{rel}/episode_{ep}")
            os.replace(part, out)
            trim_cache()
        os.utime(out)   # newest first in trim_cache
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            return self.reply(200, b"ok", "text/plain")
        url = urllib.parse.urlsplit(self.path)
        qs = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        if url.path in ("/bag", "/bag/status"):
            try:
                if qs.get("bucket") not in BAG_ROOT or not qs.get("path"):
                    raise FileNotFoundError("give bucket and path")
                if url.path == "/bag/status":
                    return self.reply(200, json.dumps(bag_status(qs["bucket"], qs["path"])).encode(), "application/json")
                src, _ = bag_paths(qs["bucket"], qs["path"])
                body = bag_page(qs["bucket"], qs["path"], self.headers.get("X-Viewer-Host") or self.headers.get("Host", ""),
                                self.headers.get("X-Rerun-Token", ""))
                return self.reply(200, body.encode(), "text/html; charset=utf-8")
            except FileNotFoundError as e:
                return self.reply(404, f"not a .mcap bag in the bucket: {e}".encode(), "text/plain")
        b = BAG_RRD.match(urllib.parse.unquote(url.path))
        if b:
            try:
                _, out = bag_paths(b["bucket"], b["path"])
            except FileNotFoundError as e:
                return self.reply(404, f"no such bag: {e}".encode(), "text/plain")
            if not os.path.exists(out):
                return self.reply(404, b"not converted yet: open the bag's page in Browse Data", "text/plain")
            return self.send_file(out)
        d = DATASET.match(urllib.parse.unquote(url.path))
        if d:
            try:
                page = int(urllib.parse.parse_qs(url.query).get("page", ["0"])[0])
                body = dataset_page(d["bucket"], d["root"], self.headers.get("X-Viewer-Host") or self.headers.get("Host", ""),
                                    self.headers.get("X-Rerun-Token", ""), page, self.headers.get("X-Browse-Base", "/data"))
            except (FileNotFoundError, ValueError) as e:
                return self.reply(404, f"not a LeRobot dataset: {e}".encode(), "text/plain")
            except Exception as e:   # noqa: BLE001  report, keep serving
                self.log_error("dataset page failed for %s: %r", self.path, e)
                return self.reply(500, f"could not read the dataset: {e}".encode(), "text/plain")
            return self.reply(200, body, "text/html; charset=utf-8")
        m = PATH.match(self.path.split("?")[0])
        if not m:
            return self.reply(404, b"not a LeRobot episode path", "text/plain")
        try:
            path = recording(m["kind"], m["root"], int(m["ep"]))
        except FileNotFoundError as e:
            return self.reply(404, str(e).encode(), "text/plain")
        except Exception as e:   # noqa: BLE001  report, keep serving
            self.log_error("build failed for %s: %r", self.path, e)
            return self.reply(500, f"could not build the recording: {e}".encode(), "text/plain")
        self.send_file(path)

    def send_file(self, path):
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
