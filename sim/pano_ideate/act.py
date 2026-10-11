# /// script
# requires-python = ">=3.10"
# dependencies = ["moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebe", "diffusers>=0.36", "transformers>=4.57", "accelerate", "torchvision"]
# [tool.uv]
# override-dependencies = ["moderngl; sys_platform == 'never'"]   # MoGe's renderer, unused here; no aarch64 wheel
# ///
"""Render the end state of an action on the panorama: what the scene looks like after it, without the steps between.

    uv run sim/pano_splat/act.py datasets/pano_splat/<scene> "pick up the cup on the counter and open the fridge"

Needs steps 1 and 2. Source slurm.env first to use the VLM on Triton. The VLM (Qwen3.5-122B-A10B) reads the action, the panorama and step 2's object list, and names
the objects the action changes, with what each looks like afterwards: taken away ("remove": picked up, carried off,
thrown away) or a new state ("the fridge door open"). For each, a perspective view around the object goes to
Qwen-Image-Edit-2511 (Apache-2.0) with a second copy that outlines the object in red, and only the edit near that
object goes back onto the panorama: for a removal, the object's surroundings (shading-matched, with its contact
shadow) less every other movable or articulated object; for a new state, the regions that changed and touch it.
The depth of the changed pixels comes from MoGe-3 on a view around each changed region, scaled to the old
depth around it.
Writes <scene>/act/<action>/: plan.json (the objects and changes; reused on a rerun, --replan to redo), pano.jpg,
distance.npy, changed.png (the pixels that changed), before_after.jpg, crops/<n>.jpg (view before, marked, the
editor's output, after). serve.py uses the same functions for edits made in the viewer.
"""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import common

PLAN = (
    "This is a 360-degree panorama of a room. Direction 0 deg is the middle of the image; directions grow to the left "
    "and are negative to the right; 180 deg is at both edges. The objects found in it, one per line as id: label "
    "(kind), direction, distance from the camera, height of its centre above the floor:\n{objects}\n\n"
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
        """The movable and articulated objects, by id + 1, largest first so smaller ones stay visible on top (not
        the static ones: the surface under a removed object must change with it)."""
        if self._labels is None:
            self._labels = np.zeros((self.H, self.W), np.int16)
            todo = [(k, self.mask(k)) for k, o in self.objects.items() if o["kind"] != "static"]
            for k, m in sorted(todo, key=lambda t: -t[1].sum()):
                self._labels[m] = k + 1
        return self._labels

    @labels.setter
    def labels(self, value: np.ndarray):
        self._labels = value


class Models:
    """The VLM, the image editor and MoGe, each loaded on first use and kept. The VLM is the one on Triton (named
    after the repo, services/triton/README.md) when TRITON_URL is set (slurm.env) and local is False."""

    def __init__(self, vlm: str, editor: str, moge: str, vlm_device: str = "auto", editor_device: str = "cuda",
                 local: bool = False):
        self.names = {"vlm": vlm, "editor": editor, "moge": moge}
        self.vlm_device, self.editor_device = vlm_device, editor_device   # editor_device also runs MoGe
        self.triton = None if local else os.environ.get("TRITON_URL")
        self._vlm = self._editor = self._template = self._moge = None

    def ask_triton(self, text: str, image: np.ndarray | None, max_new_tokens: int) -> str:
        from transformers import AutoProcessor
        if self._template is None:                    # only the chat template: the model runs on Triton
            self._template = AutoProcessor.from_pretrained(self.names["vlm"])
        return common.triton_ask(self._template, self.names["vlm"], text, image, max_new_tokens)

    def ask(self, text: str, image: np.ndarray | None = None, max_new_tokens: int = 1000) -> str:
        if self.triton:
            return self.ask_triton(text, image, max_new_tokens)
        from transformers import AutoModelForImageTextToText, AutoProcessor
        if self._vlm is None:
            self._vlm = (AutoProcessor.from_pretrained(self.names["vlm"]),
                         AutoModelForImageTextToText.from_pretrained(self.names["vlm"], dtype=torch.bfloat16,
                                                                     device_map=self.vlm_device))
        proc, vlm = self._vlm
        content = ([{"type": "image", "image": Image.fromarray(image)}] if image is not None else []) + \
                  [{"type": "text", "text": text}]
        x = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors="pt", enable_thinking=False).to(vlm.device)
        with torch.no_grad():
            y = vlm.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def drop_vlm(self):
        self._vlm = None
        torch.cuda.empty_cache()

    @property
    def editor(self):
        if self._editor is None:
            from diffusers import QwenImageEditPlusPipeline
            self._editor = QwenImageEditPlusPipeline.from_pretrained(self.names["editor"], dtype=torch.bfloat16)
            self._editor.to(self.editor_device)
            self._editor.set_progress_bar_config(disable=True)
        return self._editor

    @property
    def moge(self):
        if self._moge is None:
            self._moge = common.load_moge(self.names["moge"]).to(self.editor_device)
        return self._moge


def plan(models: Models, scene: Scene, action: str) -> list[dict]:
    """The objects a free-text action changes, and how, picked by the VLM from step 2's list."""
    lines = []
    for k, o in scene.objects.items():
        x, y, z = o["center"]
        lines.append(f"{k}: {o['label']} ({o['kind']}), {np.degrees(np.arctan2(y, x)):.0f} deg, "
                     f"{np.hypot(x, y):.1f} m, {z:.1f} m")
    text = PLAN.format(objects="\n".join(lines), action=action)
    image = cv2.resize(scene.pano, (2048, 1024), interpolation=cv2.INTER_AREA)
    changes = parse_plan(scene, models.ask(text, image))
    if not changes:                                   # now and then the answer uses other keys: ask once more
        answer = models.ask(text + "\n\nUse exactly the keys ids, object, after, state_before and state_after.", image)
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
    """A view centred on the mask, its fov `margin` times the mask's angular extent (plus 10 deg), at least `least`."""
    return camera_for(scene.dirs[mask][:: max(1, mask.sum() // 20000)] @ scene.R.T, margin, least)


def camera_for(d: np.ndarray, margin: float, least: float = 40):
    """A view centred on scene-frame directions d, its fov `margin` times their angular extent (plus 10 deg)."""
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    c = d.mean(0)
    c /= np.linalg.norm(c)
    extent = np.degrees(np.percentile(np.arccos(np.clip(d @ c, -1, 1)), 98))
    return (float(np.degrees(np.arctan2(c[1], c[0]))), float(np.degrees(np.arcsin(c[2]))),
            float(np.clip(margin * extent + 10, least, 120)))


def outline(img: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """img with obj's outline in red (dimming the rest instead washes the edit out)."""
    line = np.zeros(obj.shape, np.uint8)
    cv2.drawContours(line, cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, 1,
                     max(2, obj.shape[0] // 340))
    return np.where(line[..., None] > 0, np.array([255, 0, 0], np.uint8), img)


def edit(models: Models, images: list, prompt: str, img: np.ndarray, obj: np.ndarray, ref: np.ndarray, steps: int,
         log, name: str):
    """The editor's result for the view img (colour-matched on ref, a copied red outline painted over), its raw
    output and where it differs from img. A second seed when the object barely changed."""
    S = img.shape[0]
    for seed in (0, 1):
        res = models.editor(image=[Image.fromarray(i) for i in images], prompt=prompt, negative_prompt=" ",
                            true_cfg_scale=4.0, num_inference_steps=steps,
                            generator=torch.Generator(models.editor_device).manual_seed(seed)).images[0]
        res = np.array(res.resize((S, S), Image.LANCZOS)).astype(np.float32)
        raw = res.astype(np.uint8)
        res += (img[ref].astype(np.float32) - res[ref]).mean(0)       # the editor shifts colours slightly
        res = res.clip(0, 255)
        # The outline, copied anyway, often along the object's new shape: thin strokes of pure red the view didn't have.
        redness = lambda a: (a[..., 0] > 150) & (a[..., 1:].max(-1) < 60)
        red = (redness(res) & ~redness(img)).astype(np.uint8)
        red &= ~cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8)).astype(bool)
        if red.any():
            res = cv2.inpaint(res.astype(np.uint8), cv2.dilate(red, np.ones((7, 7), np.uint8)), 5,
                              cv2.INPAINT_TELEA).astype(np.float32)
        diff = cv2.GaussianBlur(np.abs(res - img).mean(-1), (0, 0), 2) > 25
        if np.mean(diff[obj > 0]) > 0.3:
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
    """The regions of diff within zone that touch obj, with obj."""
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
    k, part = cv2.connectedComponents(((diff & zone) | (obj > 0)).astype(np.uint8))
    return np.isin(part, np.unique(part[obj > 0])) & (part > 0)


def paste(scene: Scene, res: np.ndarray, alpha: np.ndarray, view: tuple) -> np.ndarray:
    """Blend the view res onto scene.pano with alpha; the panorama pixels that changed (alpha > 0.5)."""
    mx, my, yaw, pitch, fov = view
    H, W, S = scene.H, scene.W, res.shape[0]
    py, px = np.nonzero(cv2.dilate((alpha > 0.01).astype(np.uint8), np.ones((9, 9), np.uint8)))
    u, v = mx[py, px], my[py, px]
    rows = np.arange(max(int(v.min()) - 2, 0), min(int(v.max()) + 3, H))
    cols = np.arange(W) if u.max() - u.min() > W / 2 else np.arange(max(int(u.min()) - 2, 0), min(int(u.max()) + 3, W))
    idx = (rows[:, None] * W + cols[None]).ravel()
    vx, vy, front = common.project(scene.frame, scene.dirs.reshape(-1, 3)[idx], yaw, pitch, S, fov)
    inside = front & (vx > 0) & (vx < S - 1) & (vy > 0) & (vy < S - 1)
    idx, vx, vy = idx[inside], vx[inside], vy[inside]
    pad = -len(idx) % 4096                    # remap takes at most 32767 rows: sample through a 4096-wide grid
    grid = [np.pad(a, (0, pad)).reshape(-1, 4096) for a in (vx, vy)]
    a = cv2.remap(alpha, *grid, cv2.INTER_LINEAR).reshape(-1)[:len(idx), None]
    rgb = cv2.remap(res.astype(np.uint8), *grid, cv2.INTER_LINEAR).reshape(-1, 3)[:len(idx)]
    flat = scene.pano.reshape(-1, 3)
    flat[idx] = (flat[idx] * (1 - a) + rgb * a).astype(np.uint8)
    changed = np.zeros((H, W), bool)
    changed.reshape(-1)[idx] = a[:, 0] > 0.5
    return changed


def render(models: Models, scene: Scene, c: dict, size: int = 1024, steps: int = 40, log=print):
    """Apply one change to scene.pano in place. Returns the panorama pixels that changed and the crop panels (view
    before, marked, the editor's output, after)."""
    S = size
    remove = c["after"].strip().lower().rstrip(".") == "remove"
    mask = np.any([scene.mask(i) for i in c["ids"]], 0)
    yaw, pitch, fov = crop_camera(scene, mask, 2.5 if remove else 4.0)   # an opening door swings out: more room
    mx, my = common.view(scene.frame, scene.H, scene.W, yaw, pitch, S, fov)
    img = cv2.remap(scene.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    obj = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    marked = outline(img, obj)
    change = REMOVE.format(object=c["object"]) if remove else c["after"].rstrip(".") + "."
    prompt = EDIT.format(object=c["object"], change=change[0].upper() + change[1:])

    # What goes back: for a removal, the object's box grown by a fifth of its size (its contact shadow goes too, which
    # changes too little to detect), less every other movable or articulated object; for a new state, the regions
    # that changed within the object's box grown by its full size (a door or lid swings out) and touching it.
    zone = box_zone(obj, 0.2 if remove else 1.0)
    core = cv2.dilate(obj, np.ones((15, 15), np.uint8)) > 0
    ref = ~zone if (~zone).mean() > 0.05 else cv2.dilate(obj, np.ones((61, 61), np.uint8)) == 0   # for the colours
    res, raw, diff = edit(models, [img, marked], prompt, img, obj, ref, steps, log, c["object"])
    if remove:
        lab = cv2.remap(scene.labels, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
        keep = (zone | core) & ~((lab > 0) & ~np.isin(lab, np.array(c["ids"]) + 1) & ~core)   # other objects stay
        # Match the fill's shading to its surroundings: the colour offset on the unchanged pixels around it, spread
        # smoothly across it (one global offset leaves the object's silhouette in a slightly different tone).
        ky, kx = np.nonzero(keep)
        sigma = max(np.ptp(kx), np.ptp(ky), 8) / 4
        w = (~cv2.dilate(keep.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)).astype(np.float32)
        num = cv2.GaussianBlur((img - res) * w[..., None], (0, 0), sigma)
        res = (res + num / np.maximum(cv2.GaussianBlur(w, (0, 0), sigma), 1e-3)[..., None]).clip(0, 255)
        ys, xs = np.nonzero(obj)
        feather = max(3.0, 0.06 * max(np.ptp(xs), np.ptp(ys)))
    else:
        keep, feather = changed_near(diff, zone, obj), 3.0
    alpha = np.maximum(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), feather), (keep & core).astype(np.float32))
    after = (img * (1 - alpha[..., None]) + res * alpha[..., None]).astype(np.uint8)
    changed = paste(scene, res, alpha, (mx, my, yaw, pitch, fov))
    log(f"  {c['object']}: {keep.mean() * 100:.1f}% of the view changed")
    return changed, np.hstack([img, marked, raw, after])


def cutout(scene: Scene, ids: list[int], size: int = 512) -> np.ndarray:
    """The objects as an RGBA image: a close view of them, transparent around them."""
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
    keep = np.abs(d - np.median(d)) < 0.25 * np.median(d) + 0.05
    xyz = (scene.dirs[mask][keep] * d[keep, None]) @ scene.R.T + [0, 0, scene.frame["camera_height"]]
    base = np.array([np.median(xyz[:, 0]), np.median(xyz[:, 1]), np.percentile(xyz[:, 2], 1)])
    return {"xyz": xyz.astype(np.float32), "rgb": scene.pano[mask][keep], "base": base.astype(np.float32)}


def splat(scene: Scene, xyz: np.ndarray, rgb: np.ndarray):
    """Points drawn onto the panorama where they are in front of what is there: colour, distance and mask, with
    small gaps between the points closed."""
    H, W, h = scene.H, scene.W, scene.frame["camera_height"]
    rays = xyz - [0, 0, h]
    d = np.linalg.norm(rays, axis=1)
    x, y = common.to_pixels(rays @ scene.R, H, W)                    # scene frame -> panorama frame -> pixels
    xi, yi = np.round(x).astype(int) % W, np.clip(np.round(y).astype(int), 0, H - 1)
    order = np.argsort(-d)                                            # far first: the nearest point wins
    colour, depth = np.zeros((H, W, 3), np.uint8), np.full((H, W), np.inf, np.float32)
    colour[yi[order], xi[order]], depth[yi[order], xi[order]] = rgb[order], d[order]
    hit = np.isfinite(depth)
    whole = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    gaps = (whole & ~hit).astype(np.uint8)
    if gaps.any():
        colour = cv2.inpaint(colour, gaps, 3, cv2.INPAINT_TELEA)
        depth = cv2.inpaint(np.where(hit, depth, 0).astype(np.float32), gaps, 3, cv2.INPAINT_TELEA)
    mask = whole & (depth < scene.distance - 0.02)                    # hidden behind what is nearer
    return colour, depth, mask


def place(models: Models, scene: Scene, label: str, shp: dict, rgba: np.ndarray, u: float, v: float, note: str = "",
          size: int = 1024, steps: int = 40, log=print):
    """Put an object back, from its 3D shape, standing on the surface at panorama position (u, v) (both 0-1). Its
    points, moved there and drawn from the capture point (hidden where something is in front), give where it goes, how
    large and what covers it, as an outline; the editor draws it there from its cutout (rgba), in the scene's lighting
    with a contact shadow (shown the points themselves, it keeps them as they are: they cover only the side first
    seen). Its depth is its points'. Returns the panorama pixels that changed and the new object's mask on the
    panorama, and the crop panels."""
    y, x = min(int(v * scene.H), scene.H - 1), min(int(u * scene.W), scene.W - 1)
    target = scene.R @ (scene.dirs[y, x] * scene.distance[y, x]) + [0, 0, scene.frame["camera_height"]]
    xyz = shp["xyz"] + (target - shp["base"]).astype(np.float32)
    _, depth, mask = splat(scene, xyz, shp["rgb"])
    if not mask.any():
        raise ValueError(f"the {label} would be hidden there; pick a spot in view")

    S = size
    yaw, pitch, fov = crop_camera(scene, mask, 4.0, least=20)   # small objects: a close view, or the editor misses it
    view = (*common.view(scene.frame, scene.H, scene.W, yaw, pitch, S, fov), yaw, pitch, fov)
    mx, my = view[:2]
    img = cv2.remap(scene.pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    pts = cv2.remap(mask.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    parts = cv2.findContours(cv2.dilate(pts, np.ones((9, 9), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    big = max(cv2.contourArea(c) for c in parts)
    obj = np.zeros((S, S), np.uint8)                                  # solid, however sparse the points; no strays
    cv2.fillPoly(obj, [cv2.convexHull(c) for c in parts if cv2.contourArea(c) >= 0.2 * big], 1)
    # Other objects in front of it stay as they are (not what it stands on or in: that changes with its shadow).
    lab = cv2.remap(scene.labels, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    dist = cv2.remap(scene.distance, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    front = (lab > 0) & (dist < np.median(depth[mask]) - 0.05)
    marked = outline(img, obj)
    thing = rgba[..., :3].astype(np.float32) * (rgba[..., 3:] / 255) + 235 * (1 - rgba[..., 3:] / 255)  # on light grey
    prompt = (f"Picture 2 is picture 1 with a red outline where an object goes. Picture 3 shows the {label}. Edit "
              f"picture 1: place the {label} from picture 3 so it fills the outlined area, the same object with the same "
              "colours and material, in the scene's perspective and lighting, with a soft contact shadow where it "
              "stands. " + (f"{note.strip().rstrip('.')}. " if note.strip() else "")
              + "Keep everything else exactly as it is. Do not draw the red outline.")
    zone = box_zone(obj, 0.5)
    ref = ~zone if (~zone).mean() > 0.05 else cv2.dilate(obj, np.ones((61, 61), np.uint8)) == 0
    res, raw, diff = edit(models, [img, marked, thing.astype(np.uint8)], prompt, img, obj, ref, steps, log, label)
    keep = changed_near(diff, zone, obj) & ~front                     # the object and its shadow
    core = keep & (obj > 0)
    alpha = np.maximum(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 3.0), core.astype(np.float32))
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
    mx, my, yaw, pitch, fov = view
    S = m.shape[0]
    ys, xs = np.nonzero(m > 0.5)
    out = np.zeros((scene.H, scene.W), bool)
    if not len(xs):
        return out
    u, v = mx[ys, xs], my[ys, xs]
    rows = np.arange(max(int(v.min()) - 2, 0), min(int(v.max()) + 3, scene.H))
    cols = (np.arange(scene.W) if u.max() - u.min() > scene.W / 2
            else np.arange(max(int(u.min()) - 2, 0), min(int(u.max()) + 3, scene.W)))
    idx = (rows[:, None] * scene.W + cols[None]).ravel()
    vx, vy, front = common.project(scene.frame, scene.dirs.reshape(-1, 3)[idx], yaw, pitch, S, fov)
    inside = front & (vx >= 0) & (vx < S - 0.5) & (vy >= 0) & (vy < S - 0.5)
    idx, vx, vy = idx[inside], np.round(vx[inside]).astype(int), np.round(vy[inside]).astype(int)
    out.reshape(-1)[idx[m[vy, vx] > 0.5]] = True
    return out


def update_depth(models: Models, scene: Scene, changed: np.ndarray, size: int = 768):
    """scene.distance for the changed pixels: per changed region, MoGe on one perspective view around it (not the
    whole panorama), scaled to the old depth on the unchanged ring around the region."""
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
        with torch.no_grad():
            x = torch.tensor(img / 255, dtype=torch.float32, device=models.editor_device).permute(2, 0, 1)
            new = models.moge.infer(x, fov_x=fov, apply_mask=False)["points"].norm(dim=-1).float().cpu().numpy()
        # The region's and ring's panorama pixels, sampled from the view.
        ys, xs = np.nonzero(region | ring)
        idx = ys * W + xs
        vx, vy, front = common.project(scene.frame, scene.dirs.reshape(-1, 3)[idx], yaw, pitch, size, fov)
        ok = front & (vx >= 0) & (vx < size - 1) & (vy >= 0) & (vy < size - 1)
        idx, vx, vy = idx[ok], vx[ok], vy[ok]
        pad = -len(idx) % 4096                        # remap takes at most 32767 rows: sample through a 4096-wide grid
        d = cv2.remap(new, *[np.pad(a, (0, pad)).reshape(-1, 4096) for a in (vx, vy)], cv2.INTER_LINEAR)
        d = d.reshape(-1)[:len(idx)]
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
    p.add_argument("--size", type=int, default=1024, help="view size in pixels")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--vlm", default="Qwen/Qwen3.5-122B-A10B", help="on Triton if TRITON_URL is set (slurm.env)")
    p.add_argument("--local_vlm", action="store_true", help="load the VLM here instead (bf16, ~245 GB: one B300)")
    p.add_argument("--editor", default="Qwen/Qwen-Image-Edit-2511", help="bf16, ~60 GB")
    p.add_argument("--model", default="Ruicheng/moge-3-vitg", help="depth of the changed pixels (MIT)")
    args = p.parse_args()
    out = args.scene / "act" / (args.out or re.sub(r"[^a-z0-9]+", "_", args.action.lower()).strip("_")[:60] or "action")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    scene, models = Scene(args.scene, args.input), Models(args.vlm, args.editor, args.model, local=args.local_vlm)
    original = scene.pano.copy()

    plan_file = out / "plan.json"
    if plan_file.exists() and not args.replan and json.loads(plan_file.read_text())["action"] == args.action:
        changes = json.loads(plan_file.read_text())["changes"]
    else:
        try:
            changes = plan(models, scene, args.action)
        except ValueError as e:
            raise SystemExit(str(e))
        plan_file.write_text(json.dumps({"action": args.action, "changes": changes}, indent=1))
    models.drop_vlm()
    for c in changes:
        print(f"{c['object']} ({', '.join(map(str, c['ids']))}): {c['after']}", flush=True)

    changed = np.zeros((scene.H, scene.W), bool)
    for n, c in enumerate(changes):
        mask, panels = render(models, scene, c, args.size, args.steps, log=lambda s: print(s, flush=True))
        changed |= mask
        cv2.imwrite(str(out / "crops" / f"{n}.jpg"), cv2.cvtColor(panels, cv2.COLOR_RGB2BGR))
    models._editor = None
    torch.cuda.empty_cache()
    update_depth(models, scene, changed)

    np.save(out / "distance.npy", scene.distance)
    cv2.imwrite(str(out / "pano.jpg"), cv2.cvtColor(scene.pano, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / "changed.png"), changed.astype(np.uint8) * 255)
    half = lambda a: cv2.resize(a, (scene.W // 2, scene.H // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out / "before_after.jpg"), cv2.cvtColor(np.vstack([half(original), half(scene.pano)]),
                                                            cv2.COLOR_RGB2BGR))
    print(f"{args.scene.name}: {len(changes)} objects changed ({changed.mean() * 100:.2f}% of the panorama) -> {out}")


if __name__ == "__main__":
    main()
