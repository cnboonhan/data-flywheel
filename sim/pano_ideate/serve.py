# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "trimesh", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""Walk around a panorama in the browser and change its objects: click one, say what happens to it, and the
panorama shows the end state. Runs anywhere that reaches Triton, a laptop included: every model runs on the server
(models.py), this process only does the geometry and keeps the scenes.

    uv run sim/pano_ideate/serve.py datasets/pano_ideate        # then open http://localhost:8765

Serves viewer.html and, per scene in the folder (step 1 done), the current panorama, its depth and a map of its
objects. Add a panorama from the viewer: it runs step 1 (depth.py; ~2 min, mostly on the server) and opens it with no
objects. The viewer builds a mesh from the depth (leaving out triangles across depth jumps, a far sphere of the
panorama behind them) and keeps the eye within --radius of the capture point, at the capture height, marked by a
circle on the floor: a single panorama shows nothing behind objects, so further away holes and stretched surfaces
appear. Objects are found by hand (the viewer's Objects tool): drag a rectangle to find the objects in it (VLM, then
SAM) or one object you name (SAM outlines it from the box); click an object to rename or remove it. That writes the
scene's objects/objects.json (a list made by objects.py first is kept as objects.detected.json); an object with edits
can be renamed but not removed.

Hovering highlights an object; clicking it asks what happens to it, recorded as said (the action), and how: "Take into
the inventory" erases it (the image editor, ~30 s) into the inventory; otherwise it goes in a mode you name (open, closed, ...): a new one
is drawn from the director prompt, what the image shows ("the bin lid opens, showing an empty inside"); a mode it has
been in before puts back how it looked when last seen in it (~1 s; the viewer previews each). MoGe updates the depth
of what changed. Long operations run as jobs whose progress the viewer shows.

Each edit's result is the new current panorama. Every object has a memory: its coloured 3D points, saved when it is
picked up (it then waits in the inventory; placing it moves those points onto the surface clicked, draws their outline
from the capture point, hidden where something is in front, and the editor draws the object there; it stays the same
object), and each mode's area as last seen: pixels, depth and objects, contents included (open the bin, put a cup in,
close it, open it: the cup is there). An object with two modes is an articulated one. Edits stack; Undo and Reset go
back. The state is saved to <scene>/edits/ after every edit and picked up again on restart (pano.png, distance.npy,
labels.npy, state.json, memory/, inventory/, masks/, crops/, and steps/<n>/: the viewer's files after each edit, so
its action history can show any step). Download session packs it for the assets (session.py): the objects acted on,
where each stands, the room without them, the interactions in order; sim/asset_gen makes the objects from it,
sim/colmap_splat the background.
"""

import argparse
import json
import re
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import act
import common
import depth as depth_step
import models
import objects as objects_step
import session

p = argparse.ArgumentParser()
p.add_argument("root", type=Path, nargs="?", default=Path("datasets/pano_ideate"),
               help="folder of scenes, one folder each (source.jpg, then depth/ and objects/)")
p.add_argument("--port", type=int, default=8765)
p.add_argument("--radius", type=float, default=0.4, help="m: how far the eye may move from the capture point "
                                                       "(further away the holes behind objects show)")
p.add_argument("--grid", type=int, default=1024, help="mesh vertices around the panorama (untuned)")
p.add_argument("--jump", type=float, default=0.1, help="leave out triangles whose corner distances differ by more "
                                                     "than this fraction (edges of objects; untuned)")
p.add_argument("--steps", type=int, default=40, help="image edit steps (the editor's default, untuned)")
p.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0: reachable from other machines)")
args = p.parse_args()
VIEWER = Path(__file__).with_name("viewer.html")
triton = models.Triton()
locks: dict[str, threading.Lock] = {}
lock = lambda name: locks.setdefault(name, threading.Lock())         # one edit at a time per scene
states: dict[str, dict] = {}
log = lambda s: print(s, flush=True)


# --- Jobs: long operations run in the background and report progress ------------------------------------------------
# A job is a list of phases; a phase reports done/total when it can count, otherwise its progress is its elapsed time
# over how long it took before (EXPECTED, seconds, updated as jobs run). The starting values are rough: edit, place and
# pano_depth as the README's timings, the rest untuned.
# list, segment and save_objects: finding the objects in a dragged area (~20 s in all).
EXPECTED = {"edit": 30, "restore": 2, "depth": 3, "save": 3, "place": 35, "erase": 40, "pack": 10,
            "pano_depth": 120, "list": 12, "segment": 4, "save_objects": 4}
jobs: dict[str, "Job"] = {}


class Job:
    def __init__(self, title: str, phases: list[str]):
        self.id, self.title, self.phases = uuid.uuid4().hex[:12], title, list(phases)
        self.state, self.label, self.result, self.error = "running", "", None, None
        self.at, self.done, self.total, self.since = 0, None, None, time.time()
        jobs[self.id] = self

    def enter(self, key: str, label: str, done: int | None = None, total: int | None = None):
        """Start phase key (or update its count)."""
        if key not in self.phases:
            self.phases.insert(self.at + 1, key)
        k = self.phases.index(key)
        if k != self.at or self.label != label:
            if k > self.at and self.total is None:          # learn how long timed phases take
                key0 = self.phases[self.at]
                # A moving average, 30% the latest run; 10 s for a phase not in EXPECTED (both untuned).
                EXPECTED[key0] = 0.7 * EXPECTED.get(key0, 10) + 0.3 * (time.time() - self.since)
            if k != self.at:
                self.since = time.time()
            self.at, self.label = k, label
        self.done, self.total = done, total

    def progress(self) -> float:
        """The job's fraction done: a timed phase holds at 95% of its share until it ends, the job at 99% until done."""
        weights = [EXPECTED.get(k, 10) for k in self.phases]
        here = (self.done / self.total if self.total else
                min(0.95, (time.time() - self.since) / EXPECTED.get(self.phases[self.at], 10)))
        return min(0.99, (sum(weights[:self.at]) + weights[self.at] * here) / sum(weights))

    def snapshot(self) -> dict:
        return {"id": self.id, "title": self.title, "state": self.state, "label": self.label,
                "progress": 1.0 if self.state == "done" else round(self.progress(), 3),
                "result": self.result, "error": self.error}


def start(title: str, phases: list[str], work) -> Job:
    """Run work(job) in the background as a job."""
    job = Job(title, phases)

    def go():
        try:
            job.result = work(job)
            job.state = "done"
        except Exception as e:
            traceback.print_exc()
            job.state, job.error = "error", str(e)
    threading.Thread(target=go, daemon=True).start()
    return job


def add_scene(name: str, image: bytes) -> Job:
    """A new scene from a panorama: step 1, and an empty object list (objects are added in the viewer)."""
    folder = args.root / name
    pano = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if pano is None:
        raise ValueError("That file is not an image.")
    if abs(pano.shape[1] - 2 * pano.shape[0]) > 2:                    # 2:1 within 2 px, as depth.py
        raise ValueError(f"The image is {pano.shape[1]}x{pano.shape[0]}; a 360-degree panorama is twice as wide as tall.")
    folder.mkdir(parents=True)
    cv2.imwrite(str(folder / "source.jpg"), pano, [cv2.IMWRITE_JPEG_QUALITY, 95])      # quality 95: untuned

    def work(job):
        try:
            with lock(name):
                job.enter("pano_depth", "Estimating depth on the server")
                depth_step.run(folder, triton, log=log)
                (folder / "objects" / "masks").mkdir(parents=True)
                (folder / "objects" / "objects.json").write_text("[]")
        except Exception:
            shutil.rmtree(folder, ignore_errors=True)      # so the name can be used again
            raise
        return {"scene": name}
    return start(f"Adding {name}", ["pano_depth"], work)


def scene_names():
    return sorted(d.name for d in args.root.iterdir()
                  if (d / "depth" / "frame.json").exists() and (d / "objects" / "objects.json").exists())


# --- State: the scene, what was removed, the inventory, the history, and each object's memory -----------------------
# memory: one entry per physical object, {"ids", "label", "size", "status", "mode", "shape", "modes", "views"}: status
# None, "picked up" or "placed"; mode its current mode (None until it has modes); shape its 3D points when last picked
# up (xyz, rgb, base); modes each mode's area on the panorama as last seen (idx, pano, distance, labels). Saved arrays
# are never modified, only replaced.

def snapshot(st) -> dict:
    sc = st["scene"]
    return {"pano": sc.pano.copy(), "distance": sc.distance.copy(), "labels": sc.labels.copy(),
            "objects": dict(sc.objects), "removed": set(st["removed"]), "inventory": list(st["inventory"]),
            "history": list(st["history"]),
            "memory": [{**e, "ids": set(e["ids"]), "modes": dict(e["modes"])} for e in st["memory"]]}


def restore(st, s):
    sc = st["scene"]
    sc.pano, sc.distance, sc.labels, sc.objects = s["pano"], s["distance"], s["labels"], s["objects"]
    for k in ("removed", "inventory", "history", "memory"):
        st[k] = s[k]


def state(name: str) -> dict:
    if name in states:
        return states[name]
    sc = act.Scene(args.root / name)
    st = {"scene": sc, "removed": set(), "inventory": [], "history": [], "memory": [], "undo": [],
          "version": 0, "area": {}}
    st["original"] = snapshot(st)
    out = sc.path / "edits"
    if (out / "state.json").exists():                      # carry on where the last session stopped
        saved = json.loads((out / "state.json").read_text())
        sc.pano = cv2.cvtColor(cv2.imread(str(out / "pano.png")), cv2.COLOR_BGR2RGB)
        sc.distance = np.load(out / "distance.npy")
        sc.labels = np.load(out / "labels.npy")
        sc.objects = {int(k): o for k, o in saved["objects"].items()}
        st["removed"], st["inventory"], st["history"] = set(saved["removed"]), saved["inventory"], saved["history"]
        st["memory"] = session.read_memory(out, saved["memory"])
        log(f"{name}: resumed after {len(st['history'])} edits")
    states[name] = st
    return st


def save(st):
    sc = st["scene"]
    out = sc.path / "edits"
    (out / "memory").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "pano.png"), cv2.cvtColor(sc.pano, cv2.COLOR_RGB2BGR))     # lossless: edits stack up
    np.save(out / "distance.npy", sc.distance)
    np.save(out / "labels.npy", sc.labels)
    memory = []
    for n, e in enumerate(st["memory"]):
        entry = {**e, "ids": sorted(e["ids"]), "shape": None, "modes": {}}
        if e["shape"] is not None:
            entry["shape"] = f"{n}_shape.npz"
            np.savez(out / "memory" / entry["shape"], **e["shape"])
        for m, arrays in e["modes"].items():
            entry["modes"][m] = f"{n}_mode_{session.safe(m)}.npz"
            np.savez(out / "memory" / entry["modes"][m], **arrays)
        memory.append(entry)
    (out / "state.json").write_text(json.dumps({
        "objects": {str(k): o for k, o in sc.objects.items()}, "removed": sorted(st["removed"]),
        "inventory": st["inventory"], "history": st["history"], "memory": memory},
        indent=1))


def memory_of(st, ids, label, size=None):
    """The memory entry of the object with these ids (made on first use)."""
    for n, e in enumerate(st["memory"]):
        if e["ids"] & set(ids):
            e["ids"] |= set(ids)
            return n, e
    st["memory"].append({"ids": set(ids), "label": label, "size": size, "status": None, "mode": None, "shape": None,
                         "modes": {}, "views": []})
    return len(st["memory"]) - 1, st["memory"][-1]


def entry(st, k):
    """Object k's memory entry and its index, or (None, None)."""
    return next(((n, e) for n, e in enumerate(st["memory"]) if k in e["ids"]), (None, None))


def remember_view(st, n, e, ids):
    """Keep how an object looks now, as it is picked up: its 3D points (replaced) and its cutout, added to the views
    seen so far (the inventory shows the latest; the session export carries all of them to asset_gen)."""
    sc = st["scene"]
    e["shape"] = act.shape(sc, ids)
    out = sc.path / "edits" / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    cut = cv2.cvtColor(act.cutout(sc, ids), cv2.COLOR_RGBA2BGRA)
    e["views"] = e["views"] + [f"{n}_{len(e['views'])}.png"]
    cv2.imwrite(str(out / e["views"][-1]), cut)
    cv2.imwrite(str(out / f"{n}.png"), cut)


def export_session(name, job):
    """The session zip (session.py): the interactions, the objects acted on, the room without them."""
    st = state(name)
    with lock(name):
        save(st)
        stages = {"Erasing the objects": "erase", "Packing the session": "pack"}
        f = session.export(st["scene"].path, triton, log=log,
                           progress=lambda stage, done, total: job.enter(stages[stage], stage, done, total))
    return {"file": f.name}


def interactable(st):
    lab = set(np.unique(st["scene"].labels).tolist()) - {0}
    return {k: o for k, o in st["scene"].objects.items() if k + 1 in lab and k not in st["removed"]}


# --- What the viewer loads ----------------------------------------------------------------------------------------

def encode(img, ext):                                  # JPEG quality 90: untuned
    return cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == ".jpg" else [])[1].tobytes()


def files(st, what):
    sc = st["scene"]
    if what == "pano.jpg":     # the viewer's texture, 4096 x 2048 (untuned)
        return encode(cv2.cvtColor(cv2.resize(sc.pano, (4096, 2048), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                      ".jpg"), "image/jpeg"
    if what == "depth.png":    # distance in mm, 24 bits in RGB; nearest samples, since averaging across an edge
        mm = np.clip(np.round(cv2.resize(sc.distance, (args.grid, args.grid // 2), interpolation=cv2.INTER_NEAREST)
                              * 1000), 0, 2 ** 24 - 1).astype(np.uint32)   # makes a surface that isn't there
        return encode(np.stack([mm & 255, (mm >> 8) & 255, mm >> 16], -1).astype(np.uint8), ".png"), "image/png"
    if what == "ids.png":      # object id + 1 in R (high) and G (low); B 2 for an object with modes, else 1
        lab = sc.labels.astype(np.int32)
        if st["removed"]:
            lab[np.isin(lab - 1, list(st["removed"]))] = 0
        # 2048 x 1024: the resolution of the viewer's hover, click and tint map (untuned).
        lab = cv2.resize(lab.astype(np.float32), (2048, 1024), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        kinds = np.minimum(lab, 1)
        for e in st["memory"]:
            if len(e["modes"]) > 1:
                kinds[np.isin(lab, np.array(sorted(e["ids"])) + 1)] = 2
        rgb = np.stack([lab >> 8, lab & 255, kinds], -1).astype(np.uint8)
        return encode(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), ".png"), "image/png"
    if what == "meta.json":
        return json.dumps({
            "R": sc.frame["R_moge_to_level"], "camera_height": sc.frame["camera_height"], "radius": args.radius,
            "jump": args.jump, "version": st["version"], "history": st["history"],
            "inventory": [{"item": i, "label": st["memory"][i]["label"]} for i in st["inventory"]],
            "assets": len(st["memory"]) + sum(entry(st, k)[1] is None for k in interactable(st)),
            "objects": {k: {"label": o["label"],
                            **({"memory": n, "mode": e["mode"], "modes": sorted(e["modes"])} if e else {})}
                        for k, o in interactable(st).items() for n, e in [entry(st, k)]},
        }).encode(), "application/json"
    raise KeyError(what)


def keep_step(st, i):
    """What the viewer loads, as the scene is now, kept as step i: the original is 0, each edit adds one."""
    out = st["scene"].path / "edits" / "steps" / str(i)
    if not out.exists():
        out.mkdir(parents=True)
        for f in ("pano.jpg", "depth.png", "ids.png"):
            (out / f).write_bytes(files(st, f)[0])


# --- Edits ----------------------------------------------------------------------------------------------------------

def area(sc, idx) -> dict:
    """The panorama, depth and object map at pixels idx: one state of an articulated object's area."""
    return {"idx": idx, "pano": sc.pano.reshape(-1, 3)[idx].copy(), "distance": sc.distance.reshape(-1)[idx].copy(),
            "labels": sc.labels.reshape(-1)[idx].copy()}


def put(sc, a):
    sc.pano.reshape(-1, 3)[a["idx"]] = a["pano"]
    sc.distance.reshape(-1)[a["idx"]] = a["distance"]
    sc.labels.reshape(-1)[a["idx"]] = a["labels"]


def mode_name(s):
    return " ".join((s or "").strip().lower().rstrip(".").split()) or None


def act_on(name, k, kind, mode, now, action, director, agent, job):
    """An edit of object k, as the viewer says. kind "pick": erase it into the inventory. kind "mode": put it in mode,
    switching back to how it was last seen in that mode, or, for a new mode, drawing it from the director prompt; now
    names the mode it is in when it has none yet. The action is recorded as said, with where the agent (the viewer's
    eye) stood."""
    st = state(name)
    sc = st["scene"]
    o = sc.objects[k]
    with lock(name):
        n, e = memory_of(st, [k], o["label"], o["size"])
        mode = mode_name(mode)
        before = e["mode"] or mode_name(now) or "as found"
        if kind == "mode" and not mode:
            raise ValueError("Name the mode, for example open.")
        if kind == "mode" and mode != before and mode not in e["modes"] and not director:
            raise ValueError(f"'{mode}' is a new mode: say how it looks in the director prompt.")
        keep_step(st, len(st["history"]))
        st["undo"].append(snapshot(st))
        step = len(st["history"]) + 1
        c = {"object": f"the {o['label']}", "ids": [k], "labels": [o["label"]], "memory": n}
        crops = []
        (sc.path / "edits" / "crops").mkdir(parents=True, exist_ok=True)
        if kind == "pick":
            c["after"] = "remove"
            job.enter("edit", f"Erasing the {o['label']}")
            remember_view(st, n, e, [k])
            e["status"] = "picked up"
            c["position"] = [round(float(v), 3) for v in e["shape"]["base"]]
            st["inventory"] = [i for i in st["inventory"] if i != n] + [n]
            changed, panels = act.render(triton, sc, c, steps=args.steps, log=log)
            st["removed"].add(k)
            sc.labels[sc.labels == k + 1] = 0
            crops.append(f"{step}_{k}.jpg")
            cv2.imwrite(str(sc.path / "edits" / "crops" / crops[-1]), cv2.cvtColor(panels, cv2.COLOR_RGB2BGR))
            job.enter("depth", "Updating the depth")
            act.update_depth(triton, sc, changed)
        else:
            c.update(after=director or mode, state_before=before, state_after=mode)
            log(f"{name}: the {o['label']} {before} -> {mode}")
            if mode == before:                                       # already so: nothing to do
                c["restored"] = True
            elif mode in e["modes"]:
                job.enter("restore", f"Switching the {o['label']} to {mode}")
                # Been in that mode before: put back how its area looked then; the mode it leaves keeps how it looks
                # now (contents included: the cup put in the open bin is there when it is opened again).
                e["modes"][before] = area(sc, e["modes"][mode]["idx"])
                put(sc, e["modes"][mode])
                c["restored"] = True
            else:
                job.enter("edit", f"Drawing the {o['label']} {mode}")
                old = (sc.pano.copy(), sc.distance.copy(), sc.labels.copy())   # the area is known only after
                # Described in words, not outlined: given the outlined copy, the editor drew a red outline round the
                # closed office door (painted over, its edges went soft) or left the picture as it was.
                changed, panels = act.render(triton, sc, c, steps=args.steps, described=True, log=log)
                crops.append(f"{step}_{k}.jpg")
                cv2.imwrite(str(sc.path / "edits" / "crops" / crops[-1]), cv2.cvtColor(panels, cv2.COLOR_RGB2BGR))
                job.enter("depth", "Updating the depth")
                act.update_depth(triton, sc, changed)
                # What the edit changed within the object's hull is the object now: what it closes off is hidden (the
                # mode keeps the object map, so switching back brings it back). The hull, not the outline: an open
                # door's outline is its leaves and frame, open at the floor, so the doorway between is not in it.
                # The hull of its parts over 5% of the largest (specks left out; untuned).
                parts = cv2.findContours(sc.mask(k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                big = max(cv2.contourArea(p) for p in parts)
                pts = np.vstack([p for p in parts if cv2.contourArea(p) >= 0.05 * big])
                hull = np.zeros((sc.H, sc.W), np.uint8)
                if np.ptp(pts[:, 0, 0]) < sc.W / 2:                 # not across the seam (there: the outline only)
                    cv2.fillPoly(hull, [cv2.convexHull(pts)], 1)
                sc.labels[changed & ((hull > 0) | sc.mask(k))] = k + 1
                # The mode's area: every pixel the edit changed at all (colour, depth; the feathered edge of the paste
                # too) and 7 px around (untuned), with the areas of its other modes, so that every mode of an object
                # covers the same pixels and switching puts each back exactly.
                touched = (sc.pano != old[0]).any(-1) | (sc.distance != old[1]) | (sc.labels != old[2])
                idx = np.flatnonzero(cv2.dilate(touched.astype(np.uint8), np.ones((15, 15), np.uint8)))
                for a in e["modes"].values():
                    idx = np.union1d(idx, a["idx"])
                for m, a in list(e["modes"].items()):
                    e["modes"][m] = widen(a, idx, old)
                e["modes"][before] = {"idx": idx, "pano": old[0].reshape(-1, 3)[idx],
                                      "distance": old[1].reshape(-1)[idx], "labels": old[2].reshape(-1)[idx]}
                e["modes"][mode] = area(sc, idx)
            e["mode"] = mode
        job.enter("save", "Saving")
        st["history"].append({"what": "act", "edit": kind, "id": k, "label": o["label"], "action": action,
                              "director": director, "agent": agent,
                              "changes": [{f: c[f] for f in ("object", "ids", "after", "memory", "position",
                                                             "state_before", "state_after") if f in c}
                                          | {"restored": c.get("restored", False)}],
                              "crops": crops})
        keep_step(st, len(st["history"]))
        st["version"] += 1
        save(st)
    return st["history"][-1]


def delete_mode(name, n, mode):
    """Forget object n's mode (not the one it is in). Not an edit: no undo; the history keeps the steps that used it."""
    st = state(name)
    with lock(name):
        e = st["memory"][n]
        if mode not in e["modes"]:
            raise ValueError(f"The {e['label']} has no {mode} mode.")
        if mode == e["mode"]:
            raise ValueError(f"The {e['label']} is {mode} now: switch to another mode first.")
        del e["modes"][mode]
        if len(e["modes"]) == 1:                           # only the one it is in: no modes left to switch to
            e["modes"], e["mode"] = {}, None
        save(st)
        st["version"] += 1
    log(f"{name}: deleted the {e['label']}'s {mode} mode")


def widen(a, idx, old):
    """Mode area a over the pixels idx (a superset of its own): the pixels it lacks as they were (old) before the new
    mode was drawn, which is how they look in every mode the object had so far."""
    full = {"idx": idx, "pano": old[0].reshape(-1, 3)[idx].copy(), "distance": old[1].reshape(-1)[idx].copy(),
            "labels": old[2].reshape(-1)[idx].copy()}
    at = np.searchsorted(idx, a["idx"])
    for f in ("pano", "distance", "labels"):
        full[f][at] = a[f]
    return full


def mode_view(st, n, mode, height=96):
    """How object n looked when last seen in mode: its area painted into the room as it is now, cropped to it with
    20% around (untuned), `height` px tall: the viewer's mode previews."""
    sc, e = st["scene"], st["memory"][n]
    a = e["modes"][mode] if mode != e["mode"] else area(sc, e["modes"][mode]["idx"])   # the current one: as now
    img = sc.pano.copy()
    img.reshape(-1, 3)[a["idx"]] = a["pano"]
    ys, xs = np.divmod(a["idx"], sc.W)
    if np.ptp(xs) > sc.W / 2:                                      # across the seam: centre it first
        img, xs = np.roll(img, sc.W // 2, 1), (xs + sc.W // 2) % sc.W
    pad = int(0.2 * max(np.ptp(xs), np.ptp(ys))) + 4
    crop = img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * height / crop.shape[0])), height), interpolation=cv2.INTER_AREA)
    return encode(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), ".jpg")


def place_item(name, n, u, v, action, director, agent, job):
    """Put inventory item n where (u, v) is: the director prompt tells the editor how it stands there."""
    st = state(name)
    sc = st["scene"]
    e = st["memory"][n]
    with lock(name):
        keep_step(st, len(st["history"]))
        st["undo"].append(snapshot(st))
        job.enter("place", f"Placing the {e['label']}")
        rgba = cv2.cvtColor(cv2.imread(str(sc.path / "edits" / "inventory" / f"{n}.png"), cv2.IMREAD_UNCHANGED),
                            cv2.COLOR_BGRA2RGBA)
        changed, mask, panels = act.place(triton, sc, e["label"], e["shape"], rgba, u, v, director,
                                               steps=args.steps, log=log)
        if not mask.any():
            restore(st, st["undo"].pop())
            raise ValueError(f"the editor didn't draw the {e['label']} there; try again or pick another spot")
        k = max(sc.objects) + 1
        (sc.path / "edits" / "masks").mkdir(parents=True, exist_ok=True)
        path = (sc.path / "edits" / "masks" / f"{k}.png").resolve()
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        sc.objects = {**sc.objects, k: {"id": f"{k}_{e['label']}", "label": e["label"], "center": None,
                                        "size": e["size"], "mask": str(path)}}
        sc.labels[mask] = k + 1
        job.enter("save", "Saving")
        e["ids"].add(k)                                  # the same object, back in the room
        e["shape"] = act.shape(sc, [k])                  # as it looks now
        sc.objects[k]["center"] = (e["shape"]["base"] + [0, 0, e["size"][2] / 2]).round(3).tolist()
        e["status"] = "placed"
        st["inventory"] = [i for i in st["inventory"] if i != n]
        st["history"].append({"what": "place", "id": k, "label": e["label"], "memory": n,
                              "position": [round(float(v), 3) for v in e["shape"]["base"]],
                              "action": action or "place it here", "director": director, "agent": agent,
                              "changes": [],
                              "crops": [f"{len(st['history']) + 1}_{k}.jpg"]})
        (sc.path / "edits" / "crops").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sc.path / "edits" / "crops" / st["history"][-1]["crops"][0]),
                    cv2.cvtColor(panels, cv2.COLOR_RGB2BGR))
        keep_step(st, len(st["history"]))
        st["version"] += 1
        save(st)
    return st["history"][-1]


# --- Correcting the object list ------------------------------------------------------------------------------------

def area_box(sc, uv):
    """A view around the panorama points uv ([[u, v], ...], 0-1: a grid across a rectangle dragged in the viewer) and
    their bounding box in it: (view, image, box). The view: crop_camera's margin 1.5, at least 20 deg (untuned)."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    px = np.minimum((uv[:, 0] * sc.W).astype(int), sc.W - 1)
    py = np.minimum((uv[:, 1] * sc.H).astype(int), sc.H - 1)
    seed = np.zeros((sc.H, sc.W), bool)
    seed[py, px] = True
    yaw, pitch, fov = act.crop_camera(sc, seed, 1.5, least=20)
    S = 1024                                                         # the view's px, as act.render's (untuned)
    vx, vy, front = common.project(sc.frame, sc.dirs[py, px], yaw, pitch, S, fov)
    if front.sum() < 2:
        raise ValueError("Draw the box over something in the room.")
    mx, my = common.view(sc.frame, sc.H, sc.W, yaw, pitch, S, fov)
    img = cv2.remap(sc.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    box = [vx[front].min(), vy[front].min(), vx[front].max(), vy[front].max()]
    return (mx, my, yaw, pitch, fov), img, box


def base_objects(sc, objects: dict):
    """Write the corrected object list to objects/objects.json, keeping the detected one the first time."""
    f = sc.path / "objects" / "objects.json"
    if not f.with_name("objects.detected.json").exists():
        shutil.copy(f, f.with_name("objects.detected.json"))
    f.write_text(json.dumps([objects[k] for k in sorted(objects)], indent=1))


def relabel_under(st, region):
    """The object map in region, as the remaining objects' masks give it (largest first, as act.Scene's), in the
    current state (less what was removed) and in the original."""
    sc = st["scene"]
    fresh = np.zeros(int(region.sum()), np.int16)
    todo = [(k, sc.mask(k)[region]) for k in st["original"]["objects"]]
    for k, m in sorted(todo, key=lambda t: -t[1].sum()):
        fresh[m] = k + 1
    st["original"]["labels"][region] = fresh
    fresh[np.isin(fresh - 1, list(st["removed"]))] = 0
    sc.labels[region] = fresh


def add_object(st, mask, label, base, dedupe=False):
    """Add an object (its panorama mask) to the list: the current state, the original and base, the list to write.
    dedupe: skip it when an object already covers it (IoU over 0.5 with one under it; untuned). Its id, or None."""
    sc = st["scene"]
    if mask.sum() <= 50:                                             # as objects.py's smallest object
        return None
    if dedupe:
        under = np.bincount(st["original"]["labels"][mask].astype(np.int64))
        for j in np.argsort(-under)[:3]:                             # the 3 most overlapped objects (untuned)
            if j > 0 and j - 1 in sc.objects and under[j]:
                other = sc.mask(j - 1)
                if (other & mask).sum() / (other | mask).sum() > 0.5:
                    return None
    mask, lo, hi, near = objects_step.extent(mask, sc.distance, common.points(sc.distance, sc.frame))
    k = max([*sc.objects, *base], default=-1) + 1
    oid = f"{k:02d}_{'_'.join(re.findall(r'[a-z0-9]+', label))}"
    (sc.path / "objects" / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(sc.path / "objects" / "masks" / f"{oid}.png"), mask.astype(np.uint8) * 255)
    o = {"id": oid, "label": label, "center": ((lo + hi) / 2).round(3).tolist(), "size": (hi - lo).round(3).tolist(),
         "distance": round(near, 2), "views": [], "mask": f"masks/{oid}.png"}
    base[k], sc.objects = o, {**sc.objects, k: o}
    st["area"][k] = int(mask.sum())
    for labels in (sc.labels, st["original"]["labels"]):              # above larger objects, under smaller ones
        under = labels[mask].astype(np.int64)
        labels[mask] = np.where((under == 0) | (pixel_area(st, under - 1) >= st["area"][k]), k + 1, under)
    return k


def pixel_area(st, ks: np.ndarray) -> np.ndarray:
    """The objects' sizes on the panorama (mask pixels; act.Scene draws larger ones first), read once."""
    out = np.zeros(len(ks), np.int64)
    for j in np.unique(ks):
        if j >= 0 and j not in st["area"]:
            st["area"][j] = int(st["scene"].mask(j).sum()) if j in st["scene"].objects else 0
        out[ks == j] = st["area"].get(j, 0)
    return out


def objects_changed(st, base):
    st["original"]["objects"] = base
    base_objects(st["scene"], base)
    save(st)
    st["version"] += 1


def edit_objects(name, body):
    """Correct the object list by hand: add one named object in a dragged area (SAM outlines it), remove some, rename
    one. Not an edit (no undo): it changes the scene's base list, which Reset goes back to."""
    st = state(name)
    sc = st["scene"]
    op = body.get("op")
    with lock(name):
        base = dict(st["original"]["objects"])
        edited = {k for e in st["memory"] for k in e["ids"]}
        if op == "add":
            label = " ".join(str(body.get("label", "")).lower().split())
            if not label:
                raise ValueError("Name the object.")
            view, img, box = area_box(sc, body["area"])
            seg = triton.segment(img, [box])[0]
            k = add_object(st, act.view_to_pano(sc, seg.astype(np.float32), view), label, base)
            if k is None:
                raise ValueError("Nothing found there: draw the box tighter around it.")
            log(f"{name}: added {base[k]['id']}")
            result = {"id": k, "label": base[k]["label"]}
        elif op == "remove":
            ids = {int(k) for k in body.get("ids", [])} & set(base)
            if ids & edited:
                raise ValueError("Undo what you did with the " + ", ".join(sc.objects[k]["label"] for k in ids & edited)
                                 + " before removing it.")
            if not ids:
                raise ValueError("Nothing to remove.")
            gone = np.array(sorted(ids)) + 1
            region = np.isin(st["original"]["labels"], gone) | np.isin(sc.labels, gone)
            for k in ids:
                base.pop(k)
                st["area"].pop(k, None)
            sc.objects = {j: o for j, o in sc.objects.items() if j not in ids}
            st["original"]["objects"] = base
            relabel_under(st, region)
            log(f"{name}: removed objects {sorted(ids)}")
            result = {"removed": sorted(ids)}
        elif op == "rename":
            k, label = int(body["id"]), " ".join(str(body.get("label", "")).lower().split())
            if not label or k not in sc.objects:
                raise ValueError("Name the object.")
            sc.objects = {**sc.objects, k: {**sc.objects[k], "label": label}}
            if k in base:
                base[k] = {**base[k], "label": label}
            for e in st["memory"]:
                if k in e["ids"]:
                    e["label"] = label
            result = {"id": k, "label": label}
        else:
            raise ValueError(f"unknown op {op!r}")
        objects_changed(st, base)
    return result


def detect(name, area, job):
    """Find the objects in a dragged area and add the ones not in the list yet: the VLM lists what is in a view around
    it, the boxes centred inside the area are kept, SAM outlines them."""
    st = state(name)
    sc = st["scene"]
    with lock(name):
        base = dict(st["original"]["objects"])
        job.enter("list", "Listing the objects in the area")
        view, img, box = area_box(sc, area)
        inside = lambda b: box[0] <= (b[0] + b[2]) / 2 <= box[2] and box[1] <= (b[1] + b[3]) / 2 <= box[3]
        dets = [d for d in objects_step.parse(triton.vlm(objects_step.PROMPT, img, 3000), img.shape[0])
                if inside(d["box"])]
        job.enter("segment", "Outlining them")
        masks = triton.segment(img, [d["box"] for d in dets]) if dets else []
        found = [(act.view_to_pano(sc, m.astype(np.float32), view), d["label"]) for m, d in zip(masks, dets)]
        job.enter("save_objects", "Adding them")
        added = [k for m, label in sorted(found, key=lambda f: -f[0].sum())   # largest first, as act.Scene's
                 if (k := add_object(st, m, label, base, dedupe=True)) is not None]
        objects_changed(st, base)
        log(f"{name}: found {len(found)} in the area, added {len(added)}")
    return {"found": len(found), "added": [base[k]["label"] for k in added]}


def back(name, reset):
    st = state(name)
    with lock(name):
        if reset:
            restore(st, st["original"])
            st["original"] = snapshot(st)
            st["undo"] = []
            shutil.rmtree(st["scene"].path / "edits", ignore_errors=True)
        elif st["undo"]:
            shutil.rmtree(st["scene"].path / "edits" / "steps" / str(len(st["history"])), ignore_errors=True)
            restore(st, st["undo"].pop())
            save(st)
        st["version"] += 1


class Handler(BaseHTTPRequestHandler):
    def reply(self, body: bytes, kind: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def fail(self, e, code=500):
        self.reply(json.dumps({"error": str(e)}).encode(), "application/json", code)

    def do_GET(self):
        path = self.path.split("?")[0].strip("/").split("/")
        try:
            if path == [""]:
                return self.reply(VIEWER.read_bytes(), "text/html; charset=utf-8")
            if path == ["api", "scenes"]:
                return self.reply(json.dumps(scene_names()).encode(), "application/json")
            if len(path) == 3 and path[:2] == ["api", "jobs"] and path[2] in jobs:
                return self.reply(json.dumps(jobs[path[2]].snapshot()).encode(), "application/json")
            if len(path) >= 3 and path[0] == "api" and path[1] in scene_names():
                st = state(path[1])
                edits = st["scene"].path / "edits"
                if len(path) == 4 and path[2] == "item":
                    f = edits / "inventory" / f"{int(path[3].split('.')[0])}.png"
                    return self.reply(f.read_bytes(), "image/png")
                if path[2:] == ["session.zip"]:
                    data = (st["scene"].path / "session.zip").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/zip")
                    self.send_header("Content-Disposition", f'attachment; filename="{path[1]}_session.zip"')
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    return self.wfile.write(data)
                if len(path) == 5 and path[2] == "mode" and path[4].endswith(".jpg"):   # mode/<memory>/<mode>.jpg
                    n, mode = int(path[3]), urllib.parse.unquote(path[4][:-4])
                    if n >= len(st["memory"]) or mode not in st["memory"][n]["modes"]:
                        return self.fail("not found", 404)
                    return self.reply(mode_view(st, n, mode), "image/jpeg")
                if len(path) == 4 and path[2] == "crop" and "/" not in path[3] and path[3].endswith(".jpg"):
                    return self.reply((edits / "crops" / path[3]).read_bytes(), "image/jpeg")
                if len(path) == 5 and path[2] == "step" and path[4] in ("pano.jpg", "depth.png", "ids.png"):
                    f = edits / "steps" / str(int(path[3])) / path[4]
                    return self.reply(f.read_bytes(), "image/jpeg" if f.suffix == ".jpg" else "image/png")
                if len(path) == 3:
                    return self.reply(*files(st, path[2]))
            self.fail("not found", 404)
        except (KeyError, ValueError, FileNotFoundError):
            self.fail("not found", 404)
        except Exception as e:
            traceback.print_exc()
            self.fail(e)

    def do_POST(self):
        path = self.path.split("?")[0].strip("/").split("/")
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if len(path) == 3 and path[:2] == ["api", "upload"]:      # a new panorama: the body is the image file
                name = re.sub(r"[^a-z0-9_-]+", "_", path[2].lower()).strip("_")[:40]
                if not name or (args.root / name).exists():
                    return self.fail(f"Pick another name: {name or 'empty'} is taken or empty.", 400)
                return self.reply(json.dumps({"job": add_scene(name, raw).id}).encode(), "application/json")
            body = json.loads(raw or b"{}")
            if len(path) != 3 or path[0] != "api" or path[1] not in scene_names():
                return self.fail("not found", 404)
            name, what = path[1], path[2]
            if what == "act":
                k, kind = int(body["id"]), body.get("edit")
                action, director = str(body.get("action", "")).strip(), str(body.get("director", "")).strip()
                if kind not in ("pick", "mode") or k not in interactable(state(name)):
                    return self.fail("Click an object, then pick it up or give it a mode.", 400)
                job = start(f"{state(name)['scene'].objects[k]['label']}: {action or kind}", ["edit", "depth", "save"],
                            lambda job: act_on(name, k, kind, body.get("mode"), body.get("now"), action, director,
                                               body.get("agent"), job))
                return self.reply(json.dumps({"job": job.id}).encode(), "application/json")
            if what == "export":
                job = start(f"Session of {name}", ["erase", "pack"], lambda job: export_session(name, job))
                return self.reply(json.dumps({"job": job.id}).encode(), "application/json")
            if what == "place":
                n, u, v = int(body["item"]), float(body["u"]), float(body["v"])
                if n not in state(name)["inventory"]:
                    return self.fail("That item is no longer in the inventory.", 400)
                job = start("Placing", ["place", "save"],
                            lambda job: place_item(name, n, u, v, str(body.get("action", "")).strip(),
                                                   str(body.get("director", body.get("note", ""))).strip(),
                                                   body.get("agent"), job))
                return self.reply(json.dumps({"job": job.id}).encode(), "application/json")
            if what == "unmode":
                n = int(body["memory"])
                if not 0 <= n < len(state(name)["memory"]):
                    return self.fail("not found", 404)
                delete_mode(name, n, mode_name(body.get("mode")))
                return self.reply(b"{}", "application/json")
            if what == "objects":
                return self.reply(json.dumps(edit_objects(name, body)).encode(), "application/json")
            if what == "detect":
                if not body.get("area"):
                    return self.fail("Draw a box to look in.", 400)
                job = start("Finding objects", ["list", "segment", "save_objects"],
                            lambda job: detect(name, body["area"], job))
                return self.reply(json.dumps({"job": job.id}).encode(), "application/json")
            if what in ("undo", "reset"):
                back(name, what == "reset")
                return self.reply(b"{}", "application/json")
            self.fail("not found", 404)
        except Exception as e:
            traceback.print_exc()
            self.fail(e)

    def log_message(self, fmt, *a):
        if "POST" in (a[0] if a else ""):
            log(fmt % a)


args.root.mkdir(parents=True, exist_ok=True)
log(f"scenes in {args.root}: {', '.join(scene_names()) or 'none yet'}; models: {triton}")
log(f"open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}")
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
