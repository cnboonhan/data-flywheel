# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""A scene's session as one zip: what was interacted with and how, for sim/asset_gen (the objects) and
sim/colmap_splat (the background).

    uv run sim/pano_ideate/session.py datasets/pano_ideate/<scene>          # -> <scene>/session.zip

Reads the viewer's record (<scene>/edits/). The objects interacted with (picked up, placed, opened or closed) are
erased from the room as it is now, nearby ones together in one image edit (act.py's removal, on Triton), so the
background holds only what the simulation does not add as an asset. The zip:

    scene.json                     the scene frame (depth/frame.json): z up, metres, origin on the floor below the
                                   capture point; the panorama's size
    interactions.json              the interactions in order: what was picked up, placed, opened or closed, the asset
                                   it concerns, positions (scene frame) and states
    background/pano.jpg            the room as it is now, the interacted objects erased
    background/distance.npy        its metric distance per pixel (float32)
    room/pano.jpg                  the room as it is now, as seen in the viewer
    objects/<asset>/info.json      a movable object: label, size (m), where it was last seen, whether it is in the room
    objects/<asset>/views/<k>.png  every view of it so far (RGBA cutouts)
    objects/<asset>/shape.npz      its last-seen 3D points: xyz, rgb (scene frame), base (the point it stands on)
    articulated/<asset>/info.json  an articulated object: label, size, the states seen and the current one
    articulated/<asset>/<state>.jpg and .npz  each state's view, and its area on the panorama (idx, pano, distance)
"""

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

import act
import models


def clear(triton: models.Triton, scene: act.Scene, ids: list[int], group_fov: float = 50.0, log=print,
          progress=lambda done, total: None) -> list[list[int]]:
    """Remove the objects with these ids from scene.pano and scene.distance, grouped by direction so each image edit
    removes the ones that fit in one view."""
    dirs, area = {}, {}
    for k in ids:
        m = scene.mask(k)
        if m.any():
            d = scene.dirs[m][:: max(1, m.sum() // 2000)] @ scene.R.T
            dirs[k], area[k] = d.mean(0) / np.linalg.norm(d.mean(0)), m.sum()
    groups = []
    for k in sorted(dirs, key=lambda k: -area[k]):
        for g in groups:
            if all(np.degrees(np.arccos(np.clip(dirs[k] @ dirs[j], -1, 1))) < group_fov / 2 for j in g):
                g.append(k)
                break
        else:
            groups.append([k])
    changed = np.zeros((scene.H, scene.W), bool)
    for n, g in enumerate(groups):
        progress(n, len(groups))
        names = sorted({scene.objects[k]["label"] for k in g})
        c = {"ids": g, "labels": [scene.objects[k]["label"] for k in g], "after": "remove",
             "object": "the " + ", the ".join(names)}
        mask, _ = act.render(triton, scene, c, log=log)
        changed |= mask
        scene.labels[np.isin(scene.labels - 1, g)] = 0
    progress(len(groups), len(groups))
    if changed.any():
        act.update_depth(triton, scene, changed)
    return groups


def asset_name(n: int, label: str) -> str:
    return f"{n:02d}_{''.join(c if c.isalnum() else '_' for c in label.lower()).strip('_')}"


def export(path: Path, triton: models.Triton, out: Path | None = None, log=print,
           progress=lambda stage, done, total: None) -> Path:
    """Write the session zip (see the module docstring); out defaults to <scene>/session.zip."""
    edits = path / "edits"
    if not (edits / "state.json").exists():
        raise ValueError("Nothing to export yet: act on some objects first.")
    state = json.loads((edits / "state.json").read_text())
    scene = act.Scene(path)
    scene.pano = cv2.cvtColor(cv2.imread(str(edits / "pano.png")), cv2.COLOR_BGR2RGB)
    scene.distance = np.load(edits / "distance.npy")
    scene.labels = np.load(edits / "labels.npy")
    scene.objects = {int(k): o for k, o in state["objects"].items()}
    room = scene.pano.copy()
    memory = state["memory"]
    names = {n: asset_name(n, e["label"]) for n, e in enumerate(memory)}

    # Erase what was interacted with and is still in the room.
    present = set(np.unique(scene.labels).tolist()) - {0}
    erase = sorted({k for e in memory for k in e["ids"] if k + 1 in present})
    log(f"{path.name}: erasing {len(erase)} interacted objects")
    clear(triton, scene, erase, log=log, progress=lambda d, t: progress("Erasing the interacted objects", d, t))

    progress("Packing the session", 0, 1)
    out = out or path / "session.zip"
    frame = json.loads((path / "depth" / "frame.json").read_text())
    enc = lambda rgb: cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]

    def npy(**arrays) -> bytes:
        b = io.BytesIO()
        np.savez_compressed(b, **arrays)
        return b.getvalue()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("scene.json", json.dumps({
            "scene": path.name, "exported": time.strftime("%Y-%m-%dT%H:%M:%S"), "frame": frame,
            "panorama": {"width": scene.W, "height": scene.H},
            "assets": {"objects": [names[n] for n, e in enumerate(memory) if e["kind"] == "movable"],
                       "articulated": [names[n] for n, e in enumerate(memory) if e["kind"] == "articulated"]}},
            indent=1))
        z.writestr("interactions.json", json.dumps(interactions(state, names), indent=1))
        z.writestr("background/pano.jpg", enc(scene.pano).tobytes())
        buf = io.BytesIO()
        np.save(buf, scene.distance.astype(np.float32))
        z.writestr("background/distance.npy", buf.getvalue())
        z.writestr("room/pano.jpg", enc(room).tobytes())
        for n, e in enumerate(memory):
            base = f"{'objects' if e['kind'] == 'movable' else 'articulated'}/{names[n]}/"
            saved = {m: dict(np.load(edits / "memory" / f)) for m, f in e["saved"].items()}
            info = {"label": e["label"], "kind": e["kind"], "size_m": e["size"], "ids": e["ids"],
                    "current": e["current"]}
            if e["kind"] == "movable":
                for k, v in enumerate(e.get("views", [])):
                    z.write(edits / "inventory" / v, f"{base}views/{k}.png")
                if "shape" in saved:
                    z.writestr(base + "shape.npz", npy(**saved["shape"]))
                    info["last_seen_base"] = [round(float(v), 3) for v in saved["shape"]["base"]]
                info["in_room"] = e["current"] != "picked up"
            else:
                info["states"] = sorted(saved)
                for m, a in saved.items():
                    z.writestr(f"{base}{m}.npz", npy(**a))
                    z.writestr(f"{base}{m}.jpg", enc(state_view(room, a, scene.W)).tobytes())
            z.writestr(base + "info.json", json.dumps(info, indent=1))
    log(f"{len(memory)} assets, {len(state['history'])} interactions -> {out}")
    return out


def interactions(state: dict, names: dict) -> list[dict]:
    """The history as steps a simulation can replay: pick up, place, set an articulated object's state."""
    steps = []
    for h in state["history"]:
        if h["what"] == "place":
            steps.append({"type": "place", "asset": names.get(h.get("memory")), "label": h["label"],
                          "position": h.get("position"), "said": h["action"]})
            continue
        for c in h["changes"]:
            n = c.get("memory")
            if c["after"].strip().lower().rstrip(".") == "remove":
                steps.append({"type": "pick_up", "asset": names.get(n), "label": c["object"],
                              "position": c.get("position"), "said": h["action"]})
            elif c.get("state_after"):
                steps.append({"type": "set_state", "asset": names.get(n), "label": c["object"],
                              "from": c.get("state_before"), "to": c["state_after"], "said": h["action"],
                              "restored": c.get("restored", False)})
            else:
                steps.append({"type": "change", "label": c["object"], "after": c["after"], "said": h["action"]})
    return [{"step": i + 1, **s} for i, s in enumerate(steps)]


def state_view(room: np.ndarray, a: dict, W: int) -> np.ndarray:
    """An articulated state's area painted into the room, cropped to it."""
    img = room.copy()
    img.reshape(-1, 3)[a["idx"]] = a["pano"]
    ys, xs = np.divmod(a["idx"], W)
    pad = 40
    return img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path, help="a pano_ideate scene folder with the viewer's edits")
    p.add_argument("--out", type=Path, help="zip to write (default: <scene>/session.zip)")
    a = p.parse_args()
    export(a.scene, models.Triton(), a.out)
