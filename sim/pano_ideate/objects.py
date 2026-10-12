# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""Step 2, optional: find all the objects in a panorama at once (the viewer instead finds them per dragged rectangle).

    uv run sim/pano_ideate/objects.py datasets/pano_ideate/<scene>

Needs step 1 (depth/). The panorama is cut into perspective views (8 headings at the horizon and 45 deg down, one
straight down). A VLM (Qwen3.5-122B-A10B) lists the objects in each view with a box; LLMDet looks for every name the
VLM found in the scene, in every view, and adds what the VLM missed; SAM 2.1 turns each box into a mask (all
Apache-2.0). Masks are mapped back onto the panorama, and detections of the same object from overlapping views are
merged. Each object gets a 3D box from the depth. Every object is just an object: what can be done with it (picked
up, given modes such as open and closed) is chosen in the viewer.
Writes <scene>/objects/: objects.json, masks/<id>.png (panorama-sized), views/ (the perspective views and their
cameras, reused by later steps), labels.jpg (preview).
"""

import argparse
import json
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

import common
import models

PROMPT = (
    "Detect every physical object in this indoor photo: furniture, appliances, devices, containers and items on "
    "surfaces, including small ones such as cups, mice and cables. Skip walls, floor, ceiling, windows, curtains and "
    "ceiling lights. For each object give a short singular name and its box. Give each door, drawer and cupboard door "
    'its own entry. Answer only with a JSON list: [{"label": "<name>", "bbox_2d": [x1, y1, x2, y2]}, ...], '
    "coordinates from 0 to 1000.")


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a with every box in b (x0 y0 x1 y1)."""
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def extent(mask: np.ndarray, distance: np.ndarray, points: np.ndarray):
    """Mask (gaps under 5 px closed), 3D box (scene frame, 2nd to 98th percentile) and nearest horizontal distance
    (5th percentile) of an object, without the mask pixels that bled onto the background (further than 25% + 5 cm
    from the object's median distance). Values untuned."""
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    dist = distance[mask]
    keep = np.abs(dist - np.median(dist)) < 0.25 * np.median(dist) + 0.05
    pts = points[mask][keep]
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    return mask, lo, hi, float(np.percentile(np.linalg.norm(pts[:, :2], axis=1), 5))


def parse(answer: str, size: int) -> list[dict]:
    """The VLM's answer to PROMPT for a view of size px: [{"label", "box"}], the box in px (x0 y0 x1 y1)."""
    found, seen = [], set()
    for m in re.finditer(r"\{[^{}]*\}", answer):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        label, b = str(d.get("label", "")).lower().strip(), d.get("bbox_2d")
        if not label or not isinstance(b, list) or len(b) != 4:
            continue
        try:
            box = [float(c) * size / 1000 for c in b]                    # the prompt's 0-1000 coordinates
        except (TypeError, ValueError):
            continue
        # A repeat: the same label and box to within 8 px; a box under 4 px a side goes too (both untuned).
        key = (label, *(round(c / 8) for c in box))                       # the VLM sometimes repeats itself
        if key in seen or box[2] - box[0] < 4 or box[3] - box[1] < 4:
            continue
        seen.add(key)
        found.append({"label": label, "box": box})
    return found


def run(scene: Path, triton: models.Triton, view_size: int = 1024, batch: int = 17, redetect: bool = False,
        log=print, progress=lambda stage, done, total: None) -> list[dict]:    # defaults: the CLI's help
    """Find the objects; writes <scene>/objects/. progress(stage, done, total) reports how far it got."""
    src, out = scene / "depth", scene / "objects"
    shutil.rmtree(out / "masks", ignore_errors=True)
    (out / "masks").mkdir(parents=True)
    (out / "views").mkdir(exist_ok=True)
    pano = cv2.cvtColor(cv2.imread(str(src / "pano.jpg")), cv2.COLOR_BGR2RGB)
    distance = np.load(src / "distance.npy")
    frame = common.load_frame(scene)
    H, W = distance.shape
    points = common.points(distance, frame)

    # 17 views of 90 deg, 45 deg apart (neighbours overlap by half), none above the horizon; untuned.
    cams = [(yaw, pitch) for pitch in (0, -45) for yaw in range(0, 360, 45)] + [(0, -90)]
    grids, images = [], []
    for i, (yaw, pitch) in enumerate(cams):
        mx, my = common.view(frame, H, W, yaw, pitch, view_size)
        grids.append((mx, my))
        images.append(cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
        cv2.imwrite(str(out / "views" / f"{i:02d}.jpg"), cv2.cvtColor(images[-1], cv2.COLOR_RGB2BGR))
    (out / "views" / "cameras.json").write_text(json.dumps(
        [{"view": i, "yaw": y, "pitch": pt, "fov": 90.0, "size": view_size} for i, (y, pt) in enumerate(cams)]))

    def each(stage, fn, items):
        """fn over items, `batch` at a time on Triton, reporting progress."""
        results = [None] * len(items)
        progress(stage, 0, len(items))
        with ThreadPoolExecutor(batch) as pool:
            futures = {pool.submit(fn, it): k for k, it in enumerate(items)}
            for n, f in enumerate(as_completed(futures), 1):
                results[futures[f]] = f.result()
                progress(stage, n, len(items))
        return results

    # Detect: the VLM lists the objects in each view with their box (up to 3000 tokens); LLMDet then looks for
    # every name found anywhere in the scene, in every view, and adds the boxes the VLM missed (IoU at most 0.5 with the
    # view's boxes). Both values untuned.
    cache = out / "detections.json"
    if cache.exists() and not redetect:
        dets = json.loads(cache.read_text())
    else:
        dets, names = [], set()
        answers = each("Listing the objects in each view", lambda img: triton.vlm(PROMPT, img, 3000), images)
        for i, answer in enumerate(answers):
            found = parse(answer, view_size)
            dets += [{"view": i, **d, "source": "vlm"} for d in found]
            names |= {d["label"] for d in found}
            log(f"view {i:02d} (yaw {cams[i][0]}, pitch {cams[i][1]}): {len(found)} objects")

        labels = sorted(names)
        found = each("Looking for missed objects", lambda img: triton.detect(img, labels), images) if labels else []
        added = 0
        for i, (B, S, found_names) in enumerate(found):
            mine = np.array([d["box"] for d in dets if d["view"] == i], np.float32).reshape(-1, 4)
            for k in range(len(B)):
                if len(mine) and iou(B[k:k + 1], mine).max() > 0.5:
                    continue
                dets.append({"view": i, "box": B[k].tolist(), "label": found_names[k], "source": "llmdet"})
                added += 1
        log(f"{sum(d['source'] == 'vlm' for d in dets)} boxes from the VLM, {added} more from LLMDet")
        cache.write_text(json.dumps(dets))

    # The VLM sometimes loops, listing one label in a row of equal boxes stepping across the view: drop such runs (4 or
    # more boxes with the same top, bottom and width to within 8 px; untuned).
    run_key = lambda d: (d["view"], d["label"], round(d["box"][1] / 8), round(d["box"][3] / 8),
                         round((d["box"][2] - d["box"][0]) / 8))
    runs = Counter(run_key(d) for d in dets if d.get("source", "vlm") == "vlm")
    dets = [d for d in dets if d.get("source", "vlm") != "vlm" or runs[run_key(d)] < 4]

    # Segment each box with SAM 2.1 and map the mask onto the panorama.
    by_view = [[d for d in dets if d["view"] == i] for i in range(len(images))]
    todo = [i for i in range(len(images)) if by_view[i]]
    masks = each("Outlining the objects", lambda i: triton.segment(images[i], [d["box"] for d in by_view[i]]), todo)
    for i, ms in zip(todo, masks):
        mx, my = grids[i]
        for d, m in zip(by_view[i], ms):
            px = np.round(mx[m]).astype(int) % W, np.clip(np.round(my[m]).astype(int), 0, H - 1)
            d["pixels"] = np.unique(px[1] * W + px[0])
    progress("Merging views", 0, 1)

    # Merge detections of one object from overlapping views and detectors (over 60% of the smaller mask inside the
    # larger, and the same label or over 30% of its size, so a cat stays apart from the cat tree it sits on). Masks of
    # 50 panorama pixels or fewer are dropped. The 60% and the 50 px are untuned.
    objs = []
    for d in sorted((d for d in dets if len(d.get("pixels", ())) > 50), key=lambda d: -len(d["pixels"])):
        for o in objs:
            inter = len(np.intersect1d(o["pixels"], d["pixels"], assume_unique=True))
            same = d["label"] in o["labels"]                                  # d is the smaller
            if inter > 0.6 * len(d["pixels"]) and (same or len(d["pixels"]) > 0.3 * len(o["pixels"])):
                o["pixels"] = np.union1d(o["pixels"], d["pixels"])
                o["labels"].append(d["label"])
                o["views"].append({"view": d["view"], "box": d["box"]})
                break
        else:
            objs.append({"pixels": d["pixels"], "labels": [d["label"]], "views": [{"view": d["view"], "box": d["box"]}]})

    def extent_of(pixels):
        mask = np.zeros(H * W, bool)
        mask[pixels] = True
        return extent(mask.reshape(H, W), distance, points)

    # Second merge, in 3D: parts of one object that overlap too little in the panorama (a chair's base whose mask
    # spilled onto the floor, a part cut at a view's edge) but share a label and most of their footprint on the floor
    # (over half the smaller one's, which starts at most 10 cm above the larger one's top; untuned).
    for o in objs:
        o["lo"], o["hi"] = extent_of(o["pixels"])[1:3]
    merged = True
    while merged:
        merged = False
        for a in sorted(objs, key=lambda o: -len(o["pixels"])):
            for b in objs:
                if b is a or len(b["pixels"]) > len(a["pixels"]) or not set(a["labels"]) & set(b["labels"]):
                    continue
                ov = np.clip(np.minimum(a["hi"][:2], b["hi"][:2]) - np.maximum(a["lo"][:2], b["lo"][:2]), 0,
                             None).prod()
                if ov > 0.5 * (b["hi"][:2] - b["lo"][:2]).prod() and b["lo"][2] < a["hi"][2] + 0.1:
                    a["pixels"] = np.union1d(a["pixels"], b["pixels"])
                    for f in ("labels", "views"):
                        a[f] += b[f]
                    a["lo"], a["hi"] = extent_of(a["pixels"])[1:3]
                    objs = [x for x in objs if x is not b]
                    merged = True
                    break
            if merged:
                break
    progress("Saving the objects", 0, 1)

    records, preview = [], (pano * 0.45).astype(np.uint8)          # preview: the rest dimmed (cosmetic, untuned)
    colour = np.array([255, 140, 0])
    for k, o in enumerate(objs):
        mask, lo, hi, near = extent_of(o["pixels"])
        label = Counter(o["labels"]).most_common(1)[0][0]
        oid = f"{k:02d}_{re.sub(r'[^a-z0-9]+', '_', label).strip('_')}"
        cv2.imwrite(str(out / "masks" / f"{oid}.png"), mask.astype(np.uint8) * 255)
        records.append({"id": oid, "label": label, "center": ((lo + hi) / 2).round(3).tolist(),
                        "size": (hi - lo).round(3).tolist(), "distance": round(near, 2), "views": o["views"],
                        "mask": f"masks/{oid}.png"})
        preview[mask] = 0.55 * pano[mask] + 0.45 * colour           # tinted, with a 5 px edge (cosmetic, untuned)
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        preview[edge] = colour
    (out / "objects.json").write_text(json.dumps(records, indent=1))

    # Preview with labels (half resolution).
    preview = cv2.cvtColor(cv2.resize(preview, (W // 2, H // 2)), cv2.COLOR_RGB2BGR)
    for r in records:
        ys, xs = np.nonzero(cv2.imread(str(out / r["mask"]), cv2.IMREAD_GRAYSCALE)[::2, ::2])
        if len(xs) and xs.max() - xs.min() < W // 4:     # over half the preview wide (as across the seam): no label
            text = r["label"]
            org = (int(xs.mean()) - 4 * len(text), int(ys.mean()))
            cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "labels.jpg"), preview)
    log(f"{scene.name}: {len(records)} objects -> {out}")
    for r in records:
        log(f"  {r['id']:28s} {r['distance']:5.2f} m  size {' x '.join(f'{s:.2f}' for s in r['size'])}")
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("--view_size", type=int, default=1024, help="px, each perspective view (untuned)")
    p.add_argument("--batch", type=int, default=17, help="requests sent to Triton at once "
                                                          "(17: every view in one round)")
    p.add_argument("--redetect", action="store_true", help="ignore objects/detections.json from an earlier run")
    a = p.parse_args()
    run(a.scene, models.Triton(), a.view_size, a.batch, a.redetect)
