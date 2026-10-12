# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""A scene's session as one zip: its objects and what was done with them, for sim/asset_gen (the objects) and
sim/colmap_splat (the background).

    uv run sim/pano_ideate/session.py datasets/pano_ideate/<scene>          # -> <scene>/session.zip

Reads the viewer's record (<scene>/edits/). Every object found (the viewer's Objects stage) is an asset, acted on or
not: they are erased from the room as it is now, nearby ones together in one image edit (act.py's removal, on Triton), so the background holds only what the
simulation adds as an asset. The zip:

    scene.json                     the scene frame (depth/frame.json): z up, metres, origin on the floor below the
                                   capture point; the panorama's size; the assets
    interactions.json              the interactions in order: what was picked up, placed or put in which mode, the
                                   asset it concerns, positions (scene frame), the action said, the director prompt
                                   and where the agent stood
    layout.json                    where each asset is now, from the origin (scene frame, metres): x, y of its centre
                                   on the floor plane, null for one picked up and not placed again (then where it was
                                   last seen); no height, which one view cannot tell when its bottom is hidden
    background/pano.jpg            the room as it is now, the assets erased
    background/distance.npy        its metric distance per pixel (float32)
    room/pano.jpg                  the room as it is now, as seen in the viewer
    objects/<asset>/info.json      label, size (m), whether it is in the room, where it was last seen, its modes and
                                   the current one
    objects/<asset>/views/<k>.png  every view of it so far (RGBA cutouts; as it looks now if never picked up)
    objects/<asset>/shape.npz      its last-seen 3D points: xyz, rgb (scene frame), base (the point it stands on)
    objects/<asset>/modes/<mode>.jpg and .npz  each mode as last seen, and its area on the panorama (idx, pano,
                                   distance)
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
    removes the ones that fit in one view (each pair within group_fov / 2 of each other; 50 deg is untuned)."""
    dirs, area = {}, {}
    for k in ids:
        m = scene.mask(k)
        if m.any():
            d = scene.dirs[m][:: max(1, m.sum() // 2000)] @ scene.R.T     # up to 4000 of its pixels (untuned)
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
    n, first_retry = 0, len(groups)
    while n < len(groups):                                            # grows when a group's removal misses some
        g = groups[n]
        progress(n, len(groups))
        names = sorted({scene.objects[k]["label"] for k in g})
        c = {"ids": g, "labels": [scene.objects[k]["label"] for k in g], "after": "remove",
             "object": "the " + ", the ".join(names)}
        # Described in words rather than outlined in red: given the outline, the editor left red smears and sometimes
        # returned the outlined picture unchanged. A retry removes one object on its own, framed tight: a food
        # processor survived a 40 deg view (15 untuned).
        retry = n >= first_retry
        mask, _ = act.render(triton, scene, c, least=15 if retry else 40, described=True, log=log)
        changed |= mask
        # The editor can leave some of a group in place, swap one for a similar object (a toaster into another
        # appliance; that changes its pixels as much as removing it), or leave a red smear of the outline: the VLM
        # looks where each was, and one not bare is removed again on its own, once.
        missed = [k for k in g if still_there(triton, scene, k)]
        if missed and not retry:
            log(f"  still there: {', '.join(scene.objects[k]['label'] for k in missed)}; removing each on its own")
            groups += [[k] for k in missed]
        elif missed:
            log(f"  WARNING: the {scene.objects[missed[0]]['label']} may still be in the background")
        scene.labels[np.isin(scene.labels - 1, [k for k in g if k not in missed or retry])] = 0
        n += 1
    progress(len(groups), len(groups))
    if changed.any():
        act.update_depth(triton, scene, changed)
    return groups


def still_there(triton: models.Triton, scene: act.Scene, k: int) -> bool:
    """Whether the VLM sees anything but bare surface inside object k's old outline in the room as it is now: the
    object, another object the editor drew in its place (a toaster became another appliance), or a red smear left
    from the outline the editor was shown. The outline is drawn green so it isn't taken for a smear."""
    rgba = act.cutout(scene, [k])
    img = np.ascontiguousarray(rgba[..., :3])
    contours, _ = cv2.findContours((rgba[..., 3] > 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    answer = triton.vlm("Inside the green outline, is there only the bare surface behind (wall, floor, counter, "
                        "cabinet), or is there an object or a red or orange smear? Answer bare or object.", img, 5)
    return not answer.strip().lower().startswith("bare")


def asset_name(n: int, label: str) -> str:
    return f"{n:02d}_{safe(label.lower()).strip('_')}"


def safe(name: str) -> str:
    """A name as a file name: letters and digits kept, the rest _."""
    return "".join(c if c.isalnum() else "_" for c in name)


def read_memory(edits: Path, entries: list[dict]) -> list[dict]:
    """The viewer's memory entries (serve.py) from state.json, arrays loaded. Sessions saved before modes were named
    by hand (a kind, movable or articulated, and one "saved" dict) are read as the same."""
    load = lambda f: dict(np.load(edits / "memory" / f))
    out = []
    for e in entries:
        e = {**e, "ids": set(e["ids"])}
        if "kind" in e:                                              # the old format
            saved, movable, current = e.pop("saved"), e.pop("kind") == "movable", e.pop("current", None)
            e.update(shape=saved.get("shape") if movable else None, modes={} if movable else saved,
                     status=current if movable else None, mode=None if movable else current)
        e["shape"] = load(e["shape"]) if e.get("shape") else None
        e["modes"] = {m: load(f) for m, f in e.get("modes", {}).items()}
        e.setdefault("views", [])
        out.append(e)
    return out


def export(path: Path, triton: models.Triton, out: Path | None = None, log=print,
           progress=lambda stage, done, total: None) -> Path:
    """Write the session zip (see the module docstring); out defaults to <scene>/session.zip."""
    edits = path / "edits"
    if not (edits / "state.json").exists():
        raise ValueError("Nothing to export yet: find some objects first.")
    state = json.loads((edits / "state.json").read_text())
    scene = act.Scene(path)
    scene.pano = cv2.cvtColor(cv2.imread(str(edits / "pano.png")), cv2.COLOR_BGR2RGB)
    scene.distance = np.load(edits / "distance.npy")
    scene.labels = np.load(edits / "labels.npy")
    scene.objects = {int(k): o for k, o in state["objects"].items()}
    room = scene.pano.copy()
    present = set(np.unique(scene.labels).tolist()) - {0}
    memory = read_memory(edits, state["memory"])
    for e in memory:
        e["views"] = [(edits / "inventory" / v).read_bytes() for v in e["views"]]
    known = {k for e in memory for k in e["ids"]}
    memory += [{"ids": {k}, "label": scene.objects[k]["label"], "size": scene.objects[k].get("size"), "status": None,
                "mode": None, "shape": None, "modes": {}, "views": []}
               for k in sorted(scene.objects) if k not in known and k + 1 in present]   # found, not acted on
    for e in memory:                     # never picked up: its points and cutout as it looks now, for asset_gen
        ids = [k for k in e["ids"] if k + 1 in present]
        if e["shape"] is None and ids:
            e["shape"] = act.shape(scene, ids)
            e["views"].append(cv2.imencode(".png", cv2.cvtColor(act.cutout(scene, ids), cv2.COLOR_RGBA2BGRA))[1]
                              .tobytes())
    names = {n: asset_name(n, e["label"]) for n, e in enumerate(memory)}
    layout = [{"asset": names[n], **where(scene, e)} for n, e in enumerate(memory)]

    # Erase the assets still in the room.
    erase = sorted({k for e in memory for k in e["ids"] if k + 1 in present})
    log(f"{path.name}: erasing {len(erase)} objects")
    clear(triton, scene, erase, log=log, progress=lambda d, t: progress("Erasing the objects", d, t))

    progress("Packing the session", 0, 1)
    out = out or path / "session.zip"
    frame = json.loads((path / "depth" / "frame.json").read_text())
    # JPEG quality 95: untuned.
    enc = lambda rgb: cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])[1]

    def npy(**arrays) -> bytes:
        b = io.BytesIO()
        np.savez_compressed(b, **arrays)
        return b.getvalue()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("scene.json", json.dumps({
            "scene": path.name, "exported": time.strftime("%Y-%m-%dT%H:%M:%S"), "frame": frame,
            "panorama": {"width": scene.W, "height": scene.H},
            "assets": {"objects": [names[n] for n in range(len(memory))]}}, indent=1))
        z.writestr("interactions.json", json.dumps(interactions(state, names), indent=1))
        z.writestr("layout.json", json.dumps({
            "frame": "z up, metres, origin on the floor below the capture point (scene.json)",
            "position": "x, y of the asset's centre (no height); null when it was picked up and not placed again",
            "assets": layout}, indent=1))
        z.writestr("background/pano.jpg", enc(scene.pano).tobytes())
        buf = io.BytesIO()
        np.save(buf, scene.distance.astype(np.float32))
        z.writestr("background/distance.npy", buf.getvalue())
        z.writestr("room/pano.jpg", enc(room).tobytes())
        for n, e in enumerate(memory):
            base = f"objects/{names[n]}/"
            info = {"label": e["label"], "size_m": e["size"], "ids": sorted(e["ids"]),
                    "in_room": e["status"] != "picked up", "modes": sorted(e["modes"]), "mode": e["mode"]}
            for k, v in enumerate(e["views"]):
                z.writestr(f"{base}views/{k}.png", v)
            if e["shape"] is not None:
                z.writestr(base + "shape.npz", npy(**e["shape"]))
                info["last_seen_base"] = [round(float(v), 3) for v in e["shape"]["base"]]
            for m, a in e["modes"].items():
                z.writestr(f"{base}modes/{safe(m)}.npz", npy(**a))
                z.writestr(f"{base}modes/{safe(m)}.jpg", enc(state_view(room, a, scene.W)).tobytes())
            z.writestr(base + "info.json", json.dumps(info, indent=1))
    log(f"{len(memory)} assets, {len(state['history'])} interactions -> {out}")
    return out


def where(scene: act.Scene, e: dict) -> dict:
    """Where an asset is now (layout.json): x, y of its centre in the scene frame, from the origin."""
    r = lambda v: [round(float(c), 3) for c in v[:2]]
    entry = {"label": e["label"], "size_m": e["size"],
             **({"mode": e["mode"]} if e["mode"] else {})}
    base = r(e["shape"]["base"]) if e["shape"] is not None else r(scene.objects[min(e["ids"])]["center"])
    if e["status"] == "picked up":
        return {**entry, "position": None, "last_seen": base}
    return {**entry, "position": base}


def interactions(state: dict, names: dict) -> list[dict]:
    """The history as steps a simulation can replay: pick up, place, set an object's mode. Each carries what was said
    (the action), the director prompt the image was edited with when given, and where the agent (the viewer's eye:
    x, y, z, yaw_deg, pitch_deg in the scene frame) stood when it was taken."""
    steps = []
    for h in state["history"]:
        said = {"said": h["action"], **({"director": h["director"]} if h.get("director") else {}),
                **({"agent": h["agent"]} if h.get("agent") else {})}
        if h["what"] == "place":
            steps.append({"type": "place", "asset": names.get(h.get("memory")), "label": h["label"],
                          "position": h.get("position"), **said})
            continue
        for c in h["changes"]:
            n = c.get("memory")
            if c["after"].strip().lower().rstrip(".") == "remove":
                steps.append({"type": "pick_up", "asset": names.get(n), "label": c["object"],
                              "position": c.get("position"), **said})
            elif c.get("state_after"):
                steps.append({"type": "set_mode", "asset": names.get(n), "label": c["object"],
                              "from": c.get("state_before"), "to": c["state_after"], **said,
                              "restored": c.get("restored", False)})
            else:
                steps.append({"type": "change", "label": c["object"], "after": c["after"], **said})
    return [{"step": i + 1, **s} for i, s in enumerate(steps)]


def state_view(room: np.ndarray, a: dict, W: int) -> np.ndarray:
    """An articulated state's area painted into the room, cropped to it."""
    img = room.copy()
    img.reshape(-1, 3)[a["idx"]] = a["pano"]
    ys, xs = np.divmod(a["idx"], W)
    pad = 40                                    # px around the area (untuned)
    return img[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path, help="a pano_ideate scene folder with the viewer's edits")
    p.add_argument("--out", type=Path, help="zip to write (default: <scene>/session.zip)")
    a = p.parse_args()
    export(a.scene, models.Triton(), a.out)
