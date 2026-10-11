# /// script
# requires-python = ">=3.10"
# dependencies = ["moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebe", "diffusers>=0.36", "transformers>=4.57", "accelerate", "torchvision"]
# [tool.uv]
# override-dependencies = ["moderngl; sys_platform == 'never'"]   # MoGe's renderer, unused here; no aarch64 wheel
# ///
"""Walk around a panorama in the browser and change its objects: click one, say what happens to it, and the
panorama shows the end state.

    uv run sim/pano_splat/serve.py datasets/pano_splat --port 8000

Needs steps 1 and 2. Serves viewer.html and, per scene, the current panorama, its depth and a map of the movable and
articulated objects. The viewer builds a mesh from the depth (leaving out triangles across depth jumps) and keeps the
eye within --radius of the capture point, at the capture height, marked by a circle on the floor: a single panorama
shows nothing behind objects, so further away holes and stretched surfaces appear. Hovering highlights an object;
clicking it asks for an action. The edit is act.py's, with its models: Qwen3.5-122B-A10B plans it from the action
and the clicked object (which objects change, all ids of each, their end state, and an articulated object's state
before and after), Qwen-Image-Edit-2511 renders it and MoGe-3 updates the depth (~30 s per edit). The VLM is the one on
Triton (source slurm.env first); the editor and MoGe load here at start and stay, on one GPU (~60 GB).

Each edit's result is the new current panorama. Every object has a memory of how it was last seen, in 3D, per mode.
A movable object has one: its coloured points, saved when it is picked up (it then waits in the inventory); placing
it moves those points onto the surface clicked, draws them from the capture point (hidden where something is in
front), and the editor blends it in; it stays the same object, so picking it up again updates its memory. An
articulated object has one mode per state (closed, open): its area's pixels, depth and objects, contents included;
going back to a state it has been in restores that instead of rendering (open the fridge, put a bowl in, close it,
open it: the bowl is there). Edits stack; Undo and Reset go back. The state is saved to <scene>/edits/ after every
edit and picked up again on restart (pano.png, distance.npy, labels.npy, state.json, memory/, inventory/, masks/,
crops/, and steps/<n>/: the viewer's files after each edit, so its Edits list can show any step).
"""

import argparse
import json
import shutil
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

import act

p = argparse.ArgumentParser()
p.add_argument("root", type=Path, help="folder of scenes (datasets/pano_splat)")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--radius", type=float, default=0.4, help="m: how far the eye may move from the capture point")
p.add_argument("--grid", type=int, default=1024, help="mesh vertices around the panorama")
p.add_argument("--jump", type=float, default=0.1, help="leave out triangles whose corner distances differ by more "
                                                     "than this fraction (edges of objects)")
p.add_argument("--steps", type=int, default=40, help="image edit steps")
p.add_argument("--vlm", default="Qwen/Qwen3.5-122B-A10B", help="on Triton if TRITON_URL is set (slurm.env)")
p.add_argument("--local_vlm", action="store_true", help="load the VLM here instead (bf16, ~245 GB: a whole GPU)")
p.add_argument("--vlm_gpu", type=int, default=1, help="with --local_vlm: its GPU (one whole GPU: split, it hit "
                                                     "NVLink errors)")
p.add_argument("--editor_gpu", type=int, default=0, help="the image editor's and MoGe's GPU")
p.add_argument("--editor", default="Qwen/Qwen-Image-Edit-2511")
p.add_argument("--model", default="Ruicheng/moge-3-vitg")
args = p.parse_args()
VIEWER = Path(__file__).with_name("viewer.html")
KIND = {"movable": 1, "articulated": 2}
SAME = {"opened": "open", "shut": "closed", "close": "closed"}
models = act.Models(args.vlm, args.editor, args.model, vlm_device=f"cuda:{args.vlm_gpu}",
                    editor_device=f"cuda:{args.editor_gpu}", local=args.local_vlm)
gpu = threading.Lock()
states: dict[str, dict] = {}
log = lambda s: print(s, flush=True)


def scene_names():
    return sorted(d.name for d in args.root.iterdir() if (d / "depth").is_dir() and (d / "objects").is_dir())


# --- State: the scene, what was removed, the inventory, the history, and each object's memory -----------------------
# memory: one entry per physical object, {"ids", "kind", "label", "size", "current", "saved": {mode: arrays}}; a
# movable object's one mode is "shape" (xyz, rgb, base), an articulated object's modes are its states (idx, pano,
# distance, labels: its area on the panorama). Saved arrays are never modified, only replaced.

def snapshot(st) -> dict:
    sc = st["scene"]
    return {"pano": sc.pano.copy(), "distance": sc.distance.copy(), "labels": sc.labels.copy(),
            "objects": dict(sc.objects), "removed": set(st["removed"]), "inventory": list(st["inventory"]),
            "history": list(st["history"]),
            "memory": [{**e, "ids": set(e["ids"]), "saved": dict(e["saved"])} for e in st["memory"]]}


def restore(st, s):
    sc = st["scene"]
    sc.pano, sc.distance, sc.labels, sc.objects = s["pano"], s["distance"], s["labels"], s["objects"]
    for k in ("removed", "inventory", "history", "memory"):
        st[k] = s[k]


def state(name: str) -> dict:
    if name in states:
        return states[name]
    sc = act.Scene(args.root / name)
    st = {"scene": sc, "removed": set(), "inventory": [], "history": [], "memory": [], "undo": [], "version": 0}
    st["original"] = snapshot(st)
    out = sc.path / "edits"
    if (out / "state.json").exists():                      # carry on where the last session stopped
        saved = json.loads((out / "state.json").read_text())
        sc.pano = cv2.cvtColor(cv2.imread(str(out / "pano.png")), cv2.COLOR_BGR2RGB)
        sc.distance = np.load(out / "distance.npy")
        sc.labels = np.load(out / "labels.npy")
        sc.objects = {int(k): o for k, o in saved["objects"].items()}
        st["removed"], st["inventory"], st["history"] = set(saved["removed"]), saved["inventory"], saved["history"]
        st["memory"] = [{**e, "ids": set(e["ids"]),
                         "saved": {m: dict(np.load(out / "memory" / f)) for m, f in e["saved"].items()}}
                        for e in saved["memory"]]
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
        files = {}
        for m, arrays in e["saved"].items():
            f = f"{n}_{''.join(c if c.isalnum() else '_' for c in m)}.npz"
            np.savez(out / "memory" / f, **arrays)
            files[m] = f
        memory.append({**e, "ids": sorted(e["ids"]), "saved": files})
    (out / "state.json").write_text(json.dumps({
        "objects": {str(k): o for k, o in sc.objects.items()}, "removed": sorted(st["removed"]),
        "inventory": st["inventory"], "history": st["history"], "memory": memory}, indent=1))


def memory_of(st, ids, kind, label, size=None):
    """The memory entry of the object with these ids (made on first use)."""
    for n, e in enumerate(st["memory"]):
        if e["ids"] & set(ids):
            e["ids"] |= set(ids)
            return n, e
    st["memory"].append({"ids": set(ids), "kind": kind, "label": label, "size": size, "current": None, "saved": {}})
    return len(st["memory"]) - 1, st["memory"][-1]


def interactable(st):
    lab = set(np.unique(st["scene"].labels).tolist()) - {0}
    return {k: o for k, o in st["scene"].objects.items() if k + 1 in lab and k not in st["removed"]}


# --- What the viewer loads ----------------------------------------------------------------------------------------

def encode(img, ext):
    return cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == ".jpg" else [])[1].tobytes()


def files(st, what):
    sc = st["scene"]
    if what == "pano.jpg":
        return encode(cv2.cvtColor(cv2.resize(sc.pano, (4096, 2048), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR),
                      ".jpg"), "image/jpeg"
    if what == "depth.png":    # distance in mm, 24 bits in RGB; nearest samples, since averaging across an edge
        mm = np.clip(np.round(cv2.resize(sc.distance, (args.grid, args.grid // 2), interpolation=cv2.INTER_NEAREST)
                              * 1000), 0, 2 ** 24 - 1).astype(np.uint32)   # makes a surface that isn't there
        return encode(np.stack([mm & 255, (mm >> 8) & 255, mm >> 16], -1).astype(np.uint8), ".png"), "image/png"
    if what == "ids.png":      # object id + 1 in R (high) and G (low), its kind in B; nothing = 0
        lab = sc.labels.astype(np.int32)
        if st["removed"]:
            lab[np.isin(lab - 1, list(st["removed"]))] = 0
        lab = cv2.resize(lab.astype(np.float32), (2048, 1024), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        kinds = np.zeros(lab.max() + 1, np.uint8)
        for k, o in sc.objects.items():
            if k + 1 < len(kinds):
                kinds[k + 1] = KIND.get(o["kind"], 0)
        rgb = np.stack([lab >> 8, lab & 255, kinds[lab]], -1).astype(np.uint8)
        return encode(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), ".png"), "image/png"
    if what == "meta.json":
        return json.dumps({
            "R": sc.frame["R_moge_to_level"], "camera_height": sc.frame["camera_height"], "radius": args.radius,
            "jump": args.jump, "version": st["version"], "history": st["history"],
            "inventory": [{"item": i, "label": st["memory"][i]["label"]} for i in st["inventory"]],
            "objects": {k: {"label": o["label"], "kind": o["kind"],
                            "mode": next((e["current"] for e in st["memory"] if k in e["ids"]), None)}
                        for k, o in interactable(st).items()},
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


def word(s):
    s = (s or "").strip().lower().rstrip(".")
    return SAME.get(s, s) or None


def act_on(name, k, action):
    st = state(name)
    sc = st["scene"]
    o = sc.objects[k]
    x, y, _ = o["center"]
    where = (f"{action} (this is about object {k}, the {o['label']} at {np.degrees(np.arctan2(y, x)):.0f} deg, "
             f"{np.hypot(x, y):.1f} m; change other objects only if the action itself changes them)")
    with gpu:
        t0 = time.time()
        changes = act.plan(models, sc, where)
        t1 = time.time()
        keep_step(st, len(st["history"]))
        st["undo"].append(snapshot(st))
        crops = []
        changed = np.zeros((sc.H, sc.W), bool)
        new_modes = []
        (sc.path / "edits" / "crops").mkdir(parents=True, exist_ok=True)
        for c in changes:
            log(f"{name}: {action!r} -> {c['object']} {c['ids']}: {c['after']} "
                f"({c.get('state_before', '')} -> {c.get('state_after', '')})")
            first = sc.objects[c["ids"][0]]
            remove = c["after"].strip().lower().rstrip(".") == "remove"
            before, after = word(c.get("state_before")), word(c.get("state_after"))
            if remove:
                # Its last look in 3D, then into the inventory.
                n, e = memory_of(st, c["ids"], "movable", first["label"], first["size"])
                e["saved"] = {"shape": act.shape(sc, c["ids"])}
                e["current"] = "picked up"
                (sc.path / "edits" / "inventory").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(sc.path / "edits" / "inventory" / f"{n}.png"),
                            cv2.cvtColor(act.cutout(sc, c["ids"]), cv2.COLOR_RGBA2BGRA))
                st["inventory"] = [i for i in st["inventory"] if i != n] + [n]
            elif after:
                n, e = memory_of(st, c["ids"], "articulated", first["label"])
                now = e["current"] or before
                if now == after:                          # already so: nothing to do
                    c["restored"] = True
                    log(f"  {c['object']}: already {after}")
                    continue
                if after in e["saved"]:
                    # Been in that state before: put back how its area looked, keeping the state it leaves.
                    if now:
                        e["saved"][now] = area(sc, e["saved"][after]["idx"])
                    put(sc, e["saved"][after])
                    e["current"] = after
                    c["restored"] = True
                    log(f"  {c['object']}: restored as {after}")
                    continue
                old = (sc.pano.copy(), sc.distance.copy(), sc.labels.copy())   # the changed area is known only after
            mask, panels = act.render(models, sc, c, steps=args.steps, log=log)
            changed |= mask
            if remove:
                st["removed"] |= set(c["ids"])
                sc.labels[np.isin(sc.labels - 1, c["ids"])] = 0
            elif after:
                idx = np.flatnonzero(cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)))
                if now:
                    e["saved"][now] = {"idx": idx, "pano": old[0].reshape(-1, 3)[idx], "distance": old[1].reshape(-1)[idx],
                                       "labels": old[2].reshape(-1)[idx]}
                e["current"] = after
                new_modes.append((e, after, idx))
            crops.append(f"{len(st['history']) + 1}_{c['ids'][0]}.jpg")
            cv2.imwrite(str(sc.path / "edits" / "crops" / crops[-1]), cv2.cvtColor(panels, cv2.COLOR_RGB2BGR))
        t2 = time.time()
        if changed.any():
            act.update_depth(models, sc, changed)
        log(f"  plan {t1 - t0:.1f} s, edit {t2 - t1:.1f} s, depth {time.time() - t2:.1f} s")
        for e, after, idx in new_modes:                  # with its new depth
            e["saved"][after] = area(sc, idx)
        st["history"].append({"what": "act", "id": k, "label": o["label"], "action": action,
                              "changes": [{"object": c["object"], "ids": c["ids"], "after": c["after"],
                                           "restored": c.get("restored", False)} for c in changes],
                              "crops": crops})
        keep_step(st, len(st["history"]))
        st["version"] += 1
        save(st)
    return st["history"][-1]


def place_item(name, n, u, v, note):
    st = state(name)
    sc = st["scene"]
    e = st["memory"][n]
    with gpu:
        keep_step(st, len(st["history"]))
        st["undo"].append(snapshot(st))
        rgba = cv2.cvtColor(cv2.imread(str(sc.path / "edits" / "inventory" / f"{n}.png"), cv2.IMREAD_UNCHANGED),
                            cv2.COLOR_BGRA2RGBA)
        changed, mask, panels = act.place(models, sc, e["label"], e["saved"]["shape"], rgba, u, v, note,
                                               steps=args.steps, log=log)
        if not mask.any():
            restore(st, st["undo"].pop())
            raise ValueError(f"the editor didn't draw the {e['label']} there; try again or pick another spot")
        k = max(sc.objects) + 1
        (sc.path / "edits" / "masks").mkdir(parents=True, exist_ok=True)
        path = (sc.path / "edits" / "masks" / f"{k}.png").resolve()
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        sc.objects = {**sc.objects, k: {"id": f"{k}_{e['label']}", "label": e["label"], "kind": "movable",
                                        "movable": True, "interactable": True,
                                        "center": None, "size": e["size"],
                                        "mask": str(path)}}
        sc.labels[mask] = k + 1
        e["ids"].add(k)                                  # the same object, back in the room
        e["saved"] = {"shape": act.shape(sc, [k])}       # as it looks now
        sc.objects[k]["center"] = (e["saved"]["shape"]["base"] + [0, 0, e["size"][2] / 2]).round(3).tolist()
        e["current"] = "placed"
        st["inventory"] = [i for i in st["inventory"] if i != n]
        st["history"].append({"what": "place", "id": k, "label": e["label"],
                              "action": "place it here" + (f" ({note})" if note else ""), "changes": [],
                              "crops": [f"{len(st['history']) + 1}_{k}.jpg"]})
        (sc.path / "edits" / "crops").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sc.path / "edits" / "crops" / st["history"][-1]["crops"][0]),
                    cv2.cvtColor(panels, cv2.COLOR_RGB2BGR))
        keep_step(st, len(st["history"]))
        st["version"] += 1
        save(st)
    return st["history"][-1]


def back(name, reset):
    st = state(name)
    with gpu:
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
            if len(path) >= 3 and path[0] == "api" and path[1] in scene_names():
                st = state(path[1])
                edits = st["scene"].path / "edits"
                if len(path) == 4 and path[2] == "item":
                    f = edits / "inventory" / f"{int(path[3].split('.')[0])}.png"
                    return self.reply(f.read_bytes(), "image/png")
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
        path = self.path.strip("/").split("/")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if len(path) != 3 or path[0] != "api" or path[1] not in scene_names():
                return self.fail("not found", 404)
            name, what = path[1], path[2]
            if what == "act":
                k, action = int(body["id"]), str(body["action"]).strip()
                if not action or k not in interactable(state(name)):
                    return self.fail("Pick an object and describe what happens to it.", 400)
                return self.reply(json.dumps(act_on(name, k, action)).encode(), "application/json")
            if what == "place":
                n, u, v = int(body["item"]), float(body["u"]), float(body["v"])
                if n not in state(name)["inventory"]:
                    return self.fail("That item is no longer in the inventory.", 400)
                return self.reply(json.dumps(place_item(name, n, u, v, str(body.get("note", "")))).encode(),
                                  "application/json")
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


def preload():
    """Load the models now, so the first edit doesn't wait for them."""
    with gpu:
        models.ask("Say ok.", max_new_tokens=2)
        models.editor
        models.moge
    log("models loaded")


threading.Thread(target=preload, daemon=True).start()
log(f"serving {', '.join(scene_names())} on port {args.port}")
ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()
