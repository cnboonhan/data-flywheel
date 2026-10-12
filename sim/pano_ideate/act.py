# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""Render the end state of an action on the panorama: what the scene looks like after it, without the steps between.

    uv run sim/pano_ideate/act.py datasets/pano_ideate/<scene> "pick up the cup on the desk and open the cupboard"

Needs steps 1 and 2, and Triton (models.py). The VLM (Qwen3.5-122B-A10B) reads the action, the panorama and step 2's
object list, and names the objects the action changes, with what each looks like afterwards: taken away ("remove": picked up, carried off,
thrown away) or a new state ("the fridge door open"). For each, a perspective view around the object goes to
Qwen-Image-Edit-2511 (Apache-2.0) with a second copy that outlines the object in red, and only the edit near that
object goes back onto the panorama: for a removal, the object's surroundings (shading-matched, with its contact
shadow) less every other object of about its size; for a new state, the regions that changed and touch it.
The depth of the changed pixels comes from MoGe-3 on a view around each changed region, scaled to the old
depth around it.
Writes <scene>/act/<action>/: plan.json (the objects and changes; reused on a rerun, --replan to redo), pano.jpg,
distance.npy, changed.png (the pixels that changed), before_after.jpg, crops/<n>.jpg (view before, marked, the
editor's output, after). serve.py uses the same functions for edits made in the viewer.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

import common
import models

PLAN = (
    "This is a 360-degree panorama of a room. Direction 0 deg is the middle of the image; directions grow to the left "
    "and are negative to the right; 180 deg is at both edges. The objects found in it, one per line as id: label, "
    "direction, distance from the camera, height of its centre above the floor:\n{objects}\n\n"
    "Action: {action}\n\n"
    "Which objects look different once the action is complete? Leave out every object the action does not change. "
    "Change exactly as many objects as the action names: \"a chair\" or \"one of the chairs\" is one chair (the "
    "nearest, unless the action says which), \"the chairs\" or \"all chairs\" is every chair. "
    "Several ids can be the same physical object seen from different angles (same label, about the same direction "
    "and distance): give all of them together. For each changed object give its ids, a short description that picks "
    "it out in a close-up photo (\"the white fridge on the left\"), and its state after the action: \"remove\" if the "
    "action takes it out of the room or into the robot's hand (picked up, carried away, thrown out), otherwise one "
    "short sentence describing how it looks afterwards (\"the fridge door is wide open, showing the shelves inside\"). "
    "An object that is only moved within the room is removed here. For an articulated object (a door, drawer, lid, "
    "appliance) also give its state before and after in one word each, such as closed and open. Answer only with a "
    'JSON list: [{{"ids": [<id>, ...], "object": "<description>", "after": "remove" or "<sentence>", '
    '"state_before": "<word>", "state_after": "<word>"}}, ...] (the two states only for articulated objects), for '
    'example: [{{"ids": [12, 40], "object": "the white fridge on the left", "after": "the fridge door is wide open, '
    'showing the shelves inside", "state_before": "closed", "state_after": "open"}}]')
EDIT = ("Picture 2 is picture 1 with {object} outlined in red. Edit picture 1: {change} Keep everything else exactly as "
        "it is: the same framing, lighting and every other object. Do not draw the red outline.")
# A single-picture edit, for when the outlined pair fails: the object named by where it is instead of by the outline.
DESCRIBE = ("Name the object or objects outlined in red and say where each is relative to what is next to it, in one "
            "short phrase (for example: the white food processor between the slow cooker and the toaster). Only the "
            "phrase.")
ONLY = ("Edit this picture: {change} Keep everything else exactly as it is: the same framing, lighting and every other "
        "object.")
REMOVE = ("remove {object} completely, with no trace, outline or shadow of it left, showing the empty surface where it "
          "stood and whatever it hid.")


class Scene:
    """A panorama with its depth, scene frame and step 2's objects; `pano` and `distance` are the current state."""

    def __init__(self, path: Path, source: str = "depth"):
        self.path = path
        self.pano = cv2.cvtColor(cv2.imread(str(path / source / "pano.jpg")), cv2.COLOR_BGR2RGB)
        self.distance = np.load(path / source / "distance.npy")
        self.frame = common.load_frame(path)
        self.objects = {int(o["id"].split("_")[0]): o
                        for o in json.loads((path / "objects" / "objects.json").read_text())}
        self.H, self.W = self.pano.shape[:2]
        self.R = np.array(self.frame["R_moge_to_level"])
        self.dirs = common.directions(self.H, self.W)
        self._labels = None

    def mask(self, k: int) -> np.ndarray:
        m = cv2.imread(str(self.path / "objects" / self.objects[k]["mask"]), cv2.IMREAD_GRAYSCALE) > 0   # or absolute
        if m.shape != (self.H, self.W):
            m = cv2.resize(m.astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST) > 0
        return m

    @property
    def labels(self) -> np.ndarray:
        """The objects, by id + 1, largest first so smaller ones stay visible on top."""
        if self._labels is None:
            self._labels = np.zeros((self.H, self.W), np.int16)
            todo = [(k, self.mask(k)) for k in self.objects]
            for k, m in sorted(todo, key=lambda t: -t[1].sum()):
                self._labels[m] = k + 1
        return self._labels

    @labels.setter
    def labels(self, value: np.ndarray):
        self._labels = value


def plan(triton: models.Triton, scene: Scene, action: str) -> list[dict]:
    """The objects a free-text action changes, and how, picked by the VLM from step 2's list."""
    lines = []
    for k, o in scene.objects.items():
        x, y, z = o["center"]
        lines.append(f"{k}: {o['label']}, {np.degrees(np.arctan2(y, x)):.0f} deg, "
                     f"{np.hypot(x, y):.1f} m, {z:.1f} m")
    text = PLAN.format(objects="\n".join(lines), action=action)
    image = cv2.resize(scene.pano, (2048, 1024), interpolation=cv2.INTER_AREA)   # the VLM's view of the room, untuned
    changes = parse_plan(scene, triton.vlm(text, image))
    if not changes:                                   # now and then the answer uses other keys: ask once more
        answer = triton.vlm(text + "\n\nUse exactly the keys ids, object, after, state_before and state_after.", image)
        changes = parse_plan(scene, answer)
        if not changes:
            raise ValueError(f"no object to change; the VLM answered: {answer}")
    return changes


def parse_plan(scene: Scene, answer: str) -> list[dict]:
    changes = []
    for m in re.finditer(r"\{[^{}]*\}", answer):
        try:
            c = json.loads(m.group(0))
        except ValueError:
            continue
        ids = [int(i) for i in c.get("ids", []) if str(i).isdigit() and int(i) in scene.objects]
        if ids and c.get("object") and c.get("after"):
            changes.append({"ids": ids, "labels": [scene.objects[i]["label"] for i in ids], "object": c["object"],
                            "after": c["after"], **{k: str(c[k]).lower().strip() for k in ("state_before", "state_after")
                                                    if c.get(k)}})
    return changes


def crop_camera(scene: Scene, mask: np.ndarray, margin: float, least: float = 40):
    """A view centred on the mask: yaw, pitch and a fov of `margin` times its angular extent plus 10 deg (at least
    `least`, at most 120), in degrees. The extent is the 98th percentile of its pixels' angles from the centre (at most
    40000 of them, every n-th). All untuned."""
    d = scene.dirs[mask][:: max(1, mask.sum() // 20000)] @ scene.R.T
    c = d.mean(0) / np.linalg.norm(d.mean(0))
    extent = np.degrees(np.percentile(np.arccos(np.clip(d @ c, -1, 1)), 98))
    return (float(np.degrees(np.arctan2(c[1], c[0]))), float(np.degrees(np.arcsin(c[2]))),
            float(np.clip(margin * extent + 10, least, 120)))


def outline(img: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """img with obj's outline in red (dimming the rest instead washes the edit out)."""
    line = np.zeros(obj.shape, np.uint8)
    cv2.drawContours(line, cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, 1,
                     max(2, obj.shape[0] // 340))                    # 3 px at 1024, untuned
    return np.where(line[..., None] > 0, np.array([255, 0, 0], np.uint8), img)


def edit(triton: models.Triton, images: list, prompt: str, img: np.ndarray, obj: np.ndarray, ref: np.ndarray,
         steps: int, log, name: str):
    """The editor's result for the view img (colour-matched on ref, a copied red outline painted over), its raw
    output and where it differs from img. A second seed when the object barely changed."""
    S = img.shape[0]
    for seed in (0, 1):
        res = cv2.resize(triton.edit(images, prompt, steps, seed), (S, S), interpolation=cv2.INTER_LANCZOS4)
        res = res.astype(np.float32)
        raw = res.astype(np.uint8)
        res += (img[ref].astype(np.float32) - res[ref]).mean(0)       # the editor shifts colours slightly
        res = res.clip(0, 255)
        # The outline, copied anyway, often along the object's new shape: thin strokes of pure red the view didn't have.
        # Pure red: R over 150, G and B under 60; red over 31 px wide stays (not a stroke); grown 3 px, then inpainted.
        # All untuned.
        redness = lambda a: (a[..., 0] > 150) & (a[..., 1:].max(-1) < 60)
        red = (redness(res) & ~redness(img)).astype(np.uint8)
        red &= ~cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8)).astype(bool)
        if red.any():
            res = cv2.inpaint(res.astype(np.uint8), cv2.dilate(red, np.ones((7, 7), np.uint8)), 5,
                              cv2.INPAINT_TELEA).astype(np.float32)
        diff = cv2.GaussianBlur(np.abs(res - img).mean(-1), (0, 0), 2) > 25   # changed: over 25 levels, untuned
        if np.mean(diff[obj > 0]) > 0.3:                                    # over 30% of the object (untuned)
            break
        log(f"  {name}: barely changed" + (", trying another seed" if seed == 0 else ", kept anyway"))
    return res, raw, diff


def box_zone(obj: np.ndarray, grow: float) -> np.ndarray:
    ys, xs = np.nonzero(obj)
    g = grow * max(np.ptp(xs), np.ptp(ys))
    zone = np.zeros(obj.shape, bool)
    zone[max(int(ys.min() - g), 0):int(ys.max() + g) + 1, max(int(xs.min() - g), 0):int(xs.max() + g) + 1] = True
    return zone


def changed_near(diff: np.ndarray, zone: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """The regions of diff within zone that touch obj, with obj (specks under 5 px dropped, gaps under 15 px closed;
    untuned)."""
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    k, part = cv2.connectedComponents(((diff & zone) | (obj > 0)).astype(np.uint8))
    return np.isin(part, np.unique(part[obj > 0])) & (part > 0)


def around(scene: Scene, view: tuple, near: np.ndarray) -> np.ndarray:
    """The panorama pixels (flat indices) in the box that the view's pixels `near` come from."""
    mx, my = view[:2]
    py, px = np.nonzero(near)
    u, v = mx[py, px], my[py, px]
    rows = np.arange(max(int(v.min()) - 2, 0), min(int(v.max()) + 3, scene.H))
    cols = (np.arange(scene.W) if u.max() - u.min() > scene.W / 2      # across the seam: every column
            else np.arange(max(int(u.min()) - 2, 0), min(int(u.max()) + 3, scene.W)))
    return (rows[:, None] * scene.W + cols[None]).ravel()


def lookup(scene: Scene, view: tuple, idx: np.ndarray, images: list[np.ndarray]):
    """The panorama pixels idx that the view sees, and each view image's values there (bilinear)."""
    yaw, pitch, fov = view[2:]
    S = images[0].shape[0]
    vx, vy, front = common.project(scene.frame, scene.dirs.reshape(-1, 3)[idx], yaw, pitch, S, fov)
    ok = front & (vx >= 0) & (vx < S - 1) & (vy >= 0) & (vy < S - 1)
    idx, n = idx[ok], ok.sum()
    pad = -n % 4096                           # remap takes at most 32767 rows: sample through a 4096-wide grid
    grid = [np.pad(a[ok], (0, pad)).reshape(-1, 4096) for a in (vx, vy)]
    return idx, [cv2.remap(im, *grid, cv2.INTER_LINEAR).reshape(n + pad, -1)[:n] for im in images]


def paste(scene: Scene, res: np.ndarray, alpha: np.ndarray, view: tuple) -> np.ndarray:
    """Blend the view res onto scene.pano with alpha; the panorama pixels that changed (alpha > 0.5)."""
    near = cv2.dilate((alpha > 0.01).astype(np.uint8), np.ones((9, 9), np.uint8))   # grown 4 px (untuned)
    idx, (a, rgb) = lookup(scene, view, around(scene, view, near), [alpha, res.astype(np.uint8)])
    flat = scene.pano.reshape(-1, 3)
    flat[idx] = (flat[idx] * (1 - a) + rgb * a).astype(np.uint8)
    changed = np.zeros((scene.H, scene.W), bool)
    changed.reshape(-1)[idx] = a[:, 0] > 0.5
    return changed


def render(triton: models.Triton, scene: Scene, c: dict, size: int = 1024, steps: int = 40, least: float = 40,
           described: bool = False, log=print):
    """Apply one change to scene.pano in place. Returns the panorama pixels that changed and the crop panels (view
    before, marked, the editor's output, after). size: the view in px; steps: the editor's (both untuned; an edit
    takes ~16 s at these, services/triton/README.md); least: the view's smallest fov in degrees (crop_camera; a
    small object in a 40 deg view can be left in place, its outline redrawn as a red box); described: the editor gets
    only the view and the VLM's description of the outlined object and where it is, not the outlined copy (for a
    retry: given the outline, the editor sometimes returns the outlined picture unchanged)."""
    S = size
    remove = c["after"].strip().lower().rstrip(".") == "remove"
    mask = np.any([scene.mask(i) for i in c["ids"]], 0)
    yaw, pitch, fov = crop_camera(scene, mask, 2.5 if remove else 4.0, least)   # an opening door swings out: more room
    mx, my = common.view(scene.frame, scene.H, scene.W, yaw, pitch, S, fov)
    img = cv2.remap(scene.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    obj = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    marked = outline(img, obj)
    change = REMOVE.format(object=c["object"]) if remove else c["after"].rstrip(".") + "."
    prompt = EDIT.format(object=c["object"], change=change[0].upper() + change[1:])

    # What goes back: for a removal, the object's box grown by a fifth of its size (its contact shadow goes too, which
    # changes too little to detect), less every other object of about its size; for a new state, the object and the
    # regions that changed within its box grown by its full size (a door or lid swings out) and touching it.
    zone = box_zone(obj, 0.2 if remove else 1.0)
    core = cv2.dilate(obj, np.ones((15, 15), np.uint8)) > 0             # the object + 7 px: unfeathered (untuned)
    # Colour reference: outside the zone or, if that is under 5% of the view, over 30 px from the object (untuned).
    ref = ~zone if (~zone).mean() > 0.05 else cv2.dilate(obj, np.ones((61, 61), np.uint8)) == 0   # for the colours
    if described:
        where = triton.vlm(DESCRIBE, marked, 40)                      # one short phrase; untuned
        log(f"  {c['object']}: described as {where!r}")
        change = REMOVE.format(object=where) if remove else f"{where}: {c['after'].rstrip('.')}."
        images, prompt = [img], ONLY.format(change=change[0].upper() + change[1:])
    else:
        images = [img, marked]
    res, raw, diff = edit(triton, images, prompt, img, obj, ref, steps, log, c["object"])
    if remove:
        lab = cv2.remap(scene.labels, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP).astype(np.int64)
        # Other objects stay, except ones over 4 times its size in the view: the desk it stands on, the wall unit
        # behind it, whose surface loses its shadow with it (4: untuned).
        big = np.bincount(lab.ravel()) > 4 * max(int((obj > 0).sum()), 1)
        other = (lab > 0) & ~np.isin(lab, np.array(c["ids"]) + 1) & ~big[lab]
        keep = (zone | core) & ~(other & ~core)
        ys, xs = np.nonzero(obj)
        feather = max(3.0, 0.06 * max(np.ptp(xs), np.ptp(ys)))        # px: 6% of the object's size (untuned)
    else:
        # The whole object goes back, not only what changed a lot: where the new look is close to the old one (frosted
        # glass on a white wall) the old one showed through, a ghost of a door leaf.
        keep, feather = changed_near(diff, zone, obj) | core, 3.0       # px, untuned
    # Match what goes back to its surroundings' shading: the colour offset on the unchanged pixels around it, spread
    # smoothly across it (one global offset leaves its outline in a slightly different tone; without it a closed door
    # sat in a lighter rectangle of wall): measured 4 px clear of the kept area, spread over a quarter of its size
    # (untuned).
    ky, kx = np.nonzero(keep)
    sigma = max(np.ptp(kx), np.ptp(ky), 8) / 4
    w = (~cv2.dilate(keep.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)).astype(np.float32)
    num = cv2.GaussianBlur((img - res) * w[..., None], (0, 0), sigma)
    res = (res + num / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-3)[..., None]).clip(0, 255)
    alpha = np.maximum(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), feather), (keep & core).astype(np.float32))
    after = (img * (1 - alpha[..., None]) + res * alpha[..., None]).astype(np.uint8)
    changed = paste(scene, res, alpha, (mx, my, yaw, pitch, fov))
    log(f"  {c['object']}: {keep.mean() * 100:.1f}% of the view changed")
    return changed, np.hstack([img, marked, raw, after])


def cutout(scene: Scene, ids: list[int], size: int = 512) -> np.ndarray:
    """The objects as an RGBA image: a close view of them, transparent around them (`size` px, crop_camera's margin
    1.3, edges softened 1 px; untuned)."""
    mask = np.any([scene.mask(i) for i in ids], 0)
    yaw, pitch, fov = crop_camera(scene, mask, 1.3)
    mx, my = common.view(scene.frame, scene.H, scene.W, yaw, pitch, size, fov)
    img = cv2.remap(scene.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    a = cv2.remap(mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return np.dstack([img, (cv2.GaussianBlur(a, (0, 0), 1) * 255).astype(np.uint8)])


def shape(scene: Scene, ids: list[int]) -> dict:
    """How the objects look now, in 3D: their pixels as coloured points (scene frame), without the mask pixels that
    bled onto the background (far from the median distance), and the point they stand on."""
    mask = np.any([scene.mask(i) for i in ids], 0) & np.isin(scene.labels, np.array(ids) + 1)   # not what covers it
    if not mask.any():
        mask = np.any([scene.mask(i) for i in ids], 0)
    d = scene.distance[mask]
    # Bled pixels: further than 25% + 5 cm from the median, as objects.py's; the base: the median x, y at the 1st
    # percentile of z. Untuned.
    keep = np.abs(d - np.median(d)) < 0.25 * np.median(d) + 0.05
    xyz = (scene.dirs[mask][keep] * d[keep, None]) @ scene.R.T + [0, 0, scene.frame["camera_height"]]
    base = np.array([np.median(xyz[:, 0]), np.median(xyz[:, 1]), np.percentile(xyz[:, 2], 1)])
    return {"xyz": xyz.astype(np.float32), "rgb": scene.pano[mask][keep], "base": base.astype(np.float32)}


def splat(scene: Scene, xyz: np.ndarray):
    """Points drawn onto the panorama, gaps between them closed (under 5 px): their distance, and where they show (over
    2 cm in front of what is there). Untuned."""
    H, W, h = scene.H, scene.W, scene.frame["camera_height"]
    rays = xyz - [0, 0, h]
    d = np.linalg.norm(rays, axis=1)
    x, y = common.to_pixels(rays @ scene.R, H, W)                    # scene frame -> panorama frame -> pixels
    xi, yi = np.round(x).astype(int) % W, np.clip(np.round(y).astype(int), 0, H - 1)
    order = np.argsort(-d)                                            # far first: the nearest point wins
    depth = np.zeros((H, W), np.float32)
    depth[yi[order], xi[order]] = d[order]
    hit = depth > 0
    whole = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    if (whole & ~hit).any():
        depth = cv2.inpaint(depth, (whole & ~hit).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return depth, whole & (depth < scene.distance - 0.02)


def place(triton: models.Triton, scene: Scene, label: str, shp: dict, rgba: np.ndarray, u: float, v: float,
          note: str = "", size: int = 1024, steps: int = 40, log=print):
    """Put an object back, from its 3D shape, standing on the surface at panorama position (u, v) (both 0-1). Its
    points, moved there and drawn from the capture point (hidden where something is in front), give where it goes, how
    large and what covers it, as an outline; the editor draws it there from its cutout (rgba), in the scene's lighting
    with a contact shadow (shown the points themselves, it keeps them as they are: they cover only the side first
    seen). Its depth is its points'. Returns the panorama pixels that changed and the new object's mask on the
    panorama, and the crop panels. size and steps as in render()."""
    y, x = min(int(v * scene.H), scene.H - 1), min(int(u * scene.W), scene.W - 1)
    target = scene.R @ (scene.dirs[y, x] * scene.distance[y, x]) + [0, 0, scene.frame["camera_height"]]
    xyz = shp["xyz"] + (target - shp["base"]).astype(np.float32)
    depth, mask = splat(scene, xyz)
    if not mask.any():
        raise ValueError(f"the {label} would be hidden there; pick a spot in view")

    S = size
    yaw, pitch, fov = crop_camera(scene, mask, 4.0, least=20)   # small objects: a close view, or the editor misses it
    view = (*common.view(scene.frame, scene.H, scene.W, yaw, pitch, S, fov), yaw, pitch, fov)
    mx, my = view[:2]
    img = cv2.remap(scene.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    pts = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    # The outline: the points joined across 8 px gaps, parts under 20% of the largest dropped (untuned).
    parts = cv2.findContours(cv2.dilate(pts, np.ones((9, 9), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big = max(cv2.contourArea(c) for c in parts)
    obj = np.zeros((S, S), np.uint8)                                  # solid, however sparse the points; no strays
    cv2.fillPoly(obj, [cv2.convexHull(c) for c in parts if cv2.contourArea(c) >= 0.2 * big], 1)
    # Other objects in front of it stay as they are (not what it stands on or in: that changes with its shadow).
    lab = cv2.remap(scene.labels, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    dist = cv2.remap(scene.distance, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    front = (lab > 0) & (dist < np.median(depth[mask]) - 0.05)        # over 5 cm nearer (untuned)
    marked = outline(img, obj)
    thing = rgba[..., :3].astype(np.float32) * (rgba[..., 3:] / 255) + 235 * (1 - rgba[..., 3:] / 255)  # on light grey
    prompt = (f"Picture 2 is picture 1 with a red outline where an object goes. Picture 3 shows the {label}. Edit "
              f"picture 1: place the {label} from picture 3 so it fills the outlined area, the same object with the same "
              "colours and material, in the scene's perspective and lighting, with a soft contact shadow where it "
              "stands. " + (f"{note.strip().rstrip('.')}. " if note.strip() else "")
              + "Keep everything else exactly as it is. Do not draw the red outline.")
    zone = box_zone(obj, 0.5)                                         # grown by half its size, for its shadow; untuned
    ref = ~zone if (~zone).mean() > 0.05 else cv2.dilate(obj, np.ones((61, 61), np.uint8)) == 0   # as in render()
    res, raw, diff = edit(triton, [img, marked, thing.astype(np.uint8)], prompt, img, obj, ref, steps, log, label)
    keep = changed_near(diff, zone, obj) & ~front                     # the object and its shadow
    core = keep & (obj > 0)
    alpha = np.maximum(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 3.0), core.astype(np.float32))  # 3 px, untuned
    after = (img * (1 - alpha[..., None]) + res * alpha[..., None]).astype(np.uint8)
    changed = paste(scene, res, alpha, view)
    # The new object: what the editor changed inside the outline. Its depth: its points', or their median between them.
    new = view_to_pano(scene, (diff & (obj > 0)).astype(np.float32), view)
    if new.any():
        near = np.isfinite(depth) & mask
        scene.distance[new] = np.where(near[new], depth[new], np.median(depth[mask]))
    log(f"  {label}: placed, {keep.mean() * 100:.1f}% of the view changed")
    return changed, new, np.hstack([img, marked, raw, after])


def view_to_pano(scene: Scene, m: np.ndarray, view: tuple) -> np.ndarray:
    """A view's mask on the panorama."""
    out = np.zeros((scene.H, scene.W), bool)
    if m.any():
        idx, (v,) = lookup(scene, view, around(scene, view, m > 0.5), [m.astype(np.float32)])
        out.reshape(-1)[idx[v[:, 0] > 0.5]] = True
    return out


def update_depth(triton: models.Triton, scene: Scene, changed: np.ndarray, size: int = 768):
    """scene.distance for the changed pixels: per changed region, MoGe on one perspective view around it (not the
    whole panorama), scaled to the old depth on the unchanged ring around the region. The ring is 20 px wide; the
    view `size` px, crop_camera's margin 1.2, at least 30 deg (all untuned)."""
    H, W = scene.H, scene.W
    changed = cv2.dilate(changed.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0    # with the blended edge
    n, regions = cv2.connectedComponents(changed.astype(np.uint8))
    depth = scene.distance.copy()
    for k in range(1, n):
        region = regions == k
        ring = (cv2.dilate(region.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0) & ~changed
        yaw, pitch, fov = crop_camera(scene, region | ring, 1.2, least=30)
        mx, my = common.view(scene.frame, H, W, yaw, pitch, size, fov)
        img = cv2.remap(scene.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        new = triton.depth(img, fov)[0]
        idx, (d,) = lookup(scene, (mx, my, yaw, pitch, fov), np.flatnonzero(region | ring), [new])
        d = d[:, 0]
        inring = ring.reshape(-1)[idx]
        if not inring.any():
            continue
        scale = np.median(scene.distance.reshape(-1)[idx[inring]] / np.maximum(d[inring], 1e-3))
        depth.reshape(-1)[idx[~inring]] = d[~inring] * scale
    scene.distance = depth.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("action", help="what happens, in words")
    p.add_argument("--input", default="depth", help="folder in the scene with the panorama to change and its depth")
    p.add_argument("--out", help="output folder in <scene>/act (default: from the action)")
    p.add_argument("--replan", action="store_true", help="ask the VLM again instead of reusing plan.json")
    p.add_argument("--size", type=int, default=1024, help="view size in pixels (untuned)")
    p.add_argument("--steps", type=int, default=40, help="image edit steps (the editor's default, untuned)")
    args = p.parse_args()
    out = args.scene / "act" / (args.out or re.sub(r"[^a-z0-9]+", "_", args.action.lower()).strip("_")[:60] or "action")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    scene, triton = Scene(args.scene, args.input), models.Triton()
    original = scene.pano.copy()

    plan_file = out / "plan.json"
    if plan_file.exists() and not args.replan and json.loads(plan_file.read_text())["action"] == args.action:
        changes = json.loads(plan_file.read_text())["changes"]
    else:
        try:
            changes = plan(triton, scene, args.action)
        except ValueError as e:
            raise SystemExit(str(e))
        plan_file.write_text(json.dumps({"action": args.action, "changes": changes}, indent=1))
    for c in changes:
        print(f"{c['object']} ({', '.join(map(str, c['ids']))}): {c['after']}", flush=True)

    changed = np.zeros((scene.H, scene.W), bool)
    for n, c in enumerate(changes):
        mask, panels = render(triton, scene, c, args.size, args.steps, log=lambda s: print(s, flush=True))
        changed |= mask
        cv2.imwrite(str(out / "crops" / f"{n}.jpg"), cv2.cvtColor(panels, cv2.COLOR_RGB2BGR))
    update_depth(triton, scene, changed)

    np.save(out / "distance.npy", scene.distance)
    # JPEG quality 95: untuned.
    cv2.imwrite(str(out / "pano.jpg"), cv2.cvtColor(scene.pano, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / "changed.png"), changed.astype(np.uint8) * 255)
    half = lambda a: cv2.resize(a, (scene.W // 2, scene.H // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "before_after.jpg"), cv2.cvtColor(np.vstack([half(original), half(scene.pano)]),
                                                            cv2.COLOR_RGB2BGR))
    print(f"{args.scene.name}: {len(changes)} objects changed ({changed.mean() * 100:.2f}% of the panorama) -> {out}")


if __name__ == "__main__":
    main()
