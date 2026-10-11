# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""Step 2: find the objects in a panorama and which are interactable: movable or articulated.

    uv run sim/pano_ideate/objects.py datasets/pano_ideate/<scene>

Needs step 1 (depth/). The panorama is cut into perspective views (8 headings at the horizon and 45 deg down, one
straight down). A VLM (Qwen3.5-122B-A10B) lists the objects in each view with a box and whether each is movable
(can be picked up, pushed or carried: chairs, bags, bins, items on surfaces), articulated (stays put, but has a part to
open or turn: doors, drawers, cupboard and appliance doors) or static (tables, desks, shelving, built-ins). LLMDet looks for every name the VLM found in the scene, in every
view, and adds what the VLM missed; SAM 2.1 turns each box into a mask (all Apache-2.0). Masks are mapped back onto
the panorama, and detections of the same object from overlapping views are merged. Each object gets a 3D box from the
depth. Each interactable candidate is checked again on its own, boxed in its largest view. Movable also needs at most
--max_size across and at least --min_thickness thick; everything else stays in the background.
Writes <scene>/objects/: objects.json, masks/<id>.png (panorama-sized), views/ (the perspective views and their
cameras, reused by later steps), labels.jpg (preview: movable objects orange, articulated cyan).
"""

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

import common
import models

MOVABLE = ("'movable' (a robot could pick it up, push it or carry it, and it is likely to be moved or taken away from "
           "one day to the next: chairs, stools, bags, backpacks, bins, boxes, cups, bottles, books, devices and other "
           "items on surfaces)")
ARTICULATED = ("'articulated' (stays in place but has a part a robot could open, close or turn: doors, cabinet and "
               "cupboard doors, drawers, oven, fridge, microwave and dishwasher doors, lids, taps, knobs and switches)")
STATIC = ("'static' (stays in place and has nothing to operate: tables, desks, counters, open shelves, sofas, beds, "
          "monitors on stands, wall-mounted or hanging decorations, steps and raised platforms, rugs and floor mats, cat "
          "trees and other pet furniture, built-in fittings, pets)")
KINDS = ("static", "articulated", "movable")
PROMPT = (
    "Detect every physical object in this indoor photo: furniture, appliances, devices, containers and items on "
    "surfaces, including small ones such as cups, mice and cables. Skip walls, floor, ceiling, windows, curtains and "
    f"ceiling lights. For each object give a short singular name, whether it is {MOVABLE}, {ARTICULATED} or {STATIC}, "
    'and its box. Give each door, drawer and cupboard door its own entry. Answer only with a JSON list: [{"label": '
    '"<name>", "kind": "movable", "articulated" or "static", "bbox_2d": [x1, y1, x2, y2]}, '
    "...], coordinates from 0 to 1000.")
CHECK = ("Look at the object inside the red box ({label}) and the room around it. Is that object {movable}, "
         "{articulated} or {static}? Answer with one word: movable, articulated or static.")


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a with every box in b (x0 y0 x1 y1)."""
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def run(scene: Path, triton: models.Triton, max_size: float = 1.5, min_thickness: float = 0.01,
        view_size: int = 1024, batch: int = 17, redetect: bool = False, log=print,    # defaults: the CLI's help
        progress=lambda stage, done, total: None) -> list[dict]:
    """Find and classify the objects; writes <scene>/objects/. progress(stage, done, total) reports how far it got."""
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

    # Detect: the VLM lists the objects in each view with their kind and box (up to 3000 tokens); LLMDet then looks for
    # every name found anywhere in the scene, in every view, and adds the boxes the VLM missed (IoU at most 0.5 with the
    # view's boxes). Both values untuned.
    cache = out / "detections.json"
    if cache.exists() and not redetect:
        dets = json.loads(cache.read_text())
    else:
        dets, kinds = [], defaultdict(Counter)
        answers = each("Listing the objects in each view", lambda img: triton.vlm(PROMPT, img, 3000), images)
        for i, answer in enumerate(answers):
            seen = set()
            for m in re.finditer(r"\{[^{}]*\}", answer):
                try:
                    d = json.loads(m.group(0))
                except ValueError:
                    continue
                label, kind, b = str(d.get("label", "")).lower().strip(), d.get("kind"), d.get("bbox_2d")
                if not label or kind not in KINDS or not isinstance(b, list) or len(b) != 4:
                    continue
                try:
                    box = [float(c) * view_size / 1000 for c in b]      # the prompt's 0-1000 coordinates
                except (TypeError, ValueError):
                    continue
                # A repeat: the same label and box to within 8 px; a box under 4 px a side goes too (both untuned).
                key = (label, *(round(c / 8) for c in box))               # the VLM sometimes repeats itself
                if key in seen or box[2] - box[0] < 4 or box[3] - box[1] < 4:
                    continue
                seen.add(key)
                dets.append({"view": i, "box": box, "label": label, "kind": kind, "source": "vlm"})
                kinds[label][kind] += 1
            log(f"view {i:02d} (yaw {cams[i][0]}, pitch {cams[i][1]}): {len(seen)} objects")

        labels = sorted(kinds)
        kind_of = {n: max(c, key=lambda k: (c[k], KINDS.index(k))) for n, c in kinds.items()}
        found = each("Looking for missed objects", lambda img: triton.detect(img, labels), images) if labels else []
        added = 0
        for i, (B, S, names) in enumerate(found):
            mine = np.array([d["box"] for d in dets if d["view"] == i], np.float32).reshape(-1, 4)
            for k in range(len(B)):
                if len(mine) and iou(B[k:k + 1], mine).max() > 0.5:
                    continue
                dets.append({"view": i, "box": B[k].tolist(), "label": names[k], "kind": kind_of[names[k]],
                             "source": "llmdet"})
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
    # larger, and the same label or over 30% of its size, so a cat stays apart from the cat tree it sits on). Views can
    # disagree on the kind; the majority wins, ties going to movable. Masks of 50 panorama pixels or fewer are dropped.
    # The 60% and the 50 px are untuned.
    objs = []
    for d in sorted((d for d in dets if len(d.get("pixels", ())) > 50), key=lambda d: -len(d["pixels"])):
        for o in objs:
            inter = len(np.intersect1d(o["pixels"], d["pixels"], assume_unique=True))
            same = d["label"] in o["labels"]                                  # d is the smaller
            if inter > 0.6 * len(d["pixels"]) and (same or len(d["pixels"]) > 0.3 * len(o["pixels"])):
                o["pixels"] = np.union1d(o["pixels"], d["pixels"])
                o["labels"].append(d["label"])
                o["kinds"].append(d["kind"])
                o["views"].append({"view": d["view"], "box": d["box"]})
                break
        else:
            objs.append({"pixels": d["pixels"], "labels": [d["label"]], "kinds": [d["kind"]],
                         "views": [{"view": d["view"], "box": d["box"]}]})

    def extent(pixels):
        """Mask (gaps under 5 px closed), 3D box (scene frame, 2nd to 98th percentile) and nearest horizontal distance
        (5th percentile) of an object, without the mask pixels that bled onto the background (further than 25% + 5 cm
        from the object's median distance). Values untuned."""
        mask = np.zeros(H * W, bool)
        mask[pixels] = True
        mask = cv2.morphologyEx(mask.reshape(H, W).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        dist = distance[mask]
        keep = np.abs(dist - np.median(dist)) < 0.25 * np.median(dist) + 0.05
        pts = points[mask][keep]
        lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
        return mask, lo, hi, float(np.percentile(np.linalg.norm(pts[:, :2], axis=1), 5))

    # Second merge, in 3D: parts of one object that overlap too little in the panorama (a chair's base whose mask
    # spilled onto the floor, a part cut at a view's edge) but share a label and most of their footprint on the floor
    # (over half the smaller one's, which starts at most 10 cm above the larger one's top; untuned).
    for o in objs:
        o["lo"], o["hi"] = extent(o["pixels"])[1:3]
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
                    for f in ("labels", "kinds", "views"):
                        a[f] += b[f]
                    a["lo"], a["hi"] = extent(a["pixels"])[1:3]
                    objs = [x for x in objs if x is not b]
                    merged = True
                    break
            if merged:
                break
    for o in objs:
        o["kind"] = max(set(o["kinds"]), key=lambda k: (o["kinds"].count(k), KINDS.index(k)))

    # Check each interactable candidate on its own: the per-view naming misjudges objects seen out of context (a
    # tabletop with items on it from above). The VLM sees the view where the object is largest, boxed in red, and
    # answers in one word (10 tokens). Cached.
    checks_file = out / "checks.json"
    checks = {} if redetect or not checks_file.exists() else json.loads(checks_file.read_text())
    todo = []
    for o in objs:
        if o["kind"] == "static":
            continue
        v = max(o["views"], key=lambda v: (v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1]))
        o["label"] = Counter(o["labels"]).most_common(1)[0][0]
        o["key"] = f"{v['view']}:{','.join(f'{c:.0f}' for c in v['box'])}:{o['label']}"
        if o["key"] not in checks:
            img = images[v["view"]].copy()
            x0, y0, x1, y1 = map(int, v["box"])
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), max(2, view_size // 256))   # 4 px at 1024, untuned
            todo.append((o, img))
    texts = [CHECK.format(label=o["label"], movable=MOVABLE, articulated=ARTICULATED, static=STATIC) for o, _ in todo]
    answers = each("Checking each object", lambda it: triton.vlm(it[1], it[0], 10), [(img, t) for (_, img), t in
                                                                                      zip(todo, texts)])
    for (o, _), answer in zip(todo, answers):
        checks[o["key"]] = next((k for k in KINDS if k in answer.lower()), o["kind"])
        if checks[o["key"]] != o["kind"]:
            log(f"check: {o['label']} {o['kind']} -> {checks[o['key']]}")
    for o in objs:
        if o["kind"] != "static":
            o["kind"] = checks[o["key"]]
    checks_file.write_text(json.dumps(checks, indent=1))
    progress("Saving the objects", 0, 1)

    records, preview = [], (pano * 0.45).astype(np.uint8)          # preview: the rest dimmed (cosmetic, untuned)
    for k, o in enumerate(objs):
        mask, lo, hi, near = extent(o["pixels"])
        size = hi - lo
        label = Counter(o["labels"]).most_common(1)[0][0]
        movable = o["kind"] == "movable" and min_thickness <= size.min() and size.max() <= max_size
        oid = f"{k:02d}_{re.sub(r'[^a-z0-9]+', '_', label).strip('_')}"
        cv2.imwrite(str(out / "masks" / f"{oid}.png"), mask.astype(np.uint8) * 255)
        interactable = movable or o["kind"] == "articulated"
        records.append({"id": oid, "label": label, "kind": o["kind"], "movable": bool(movable),
                        "interactable": bool(interactable),
                        "center": ((lo + hi) / 2).round(3).tolist(), "size": size.round(3).tolist(),
                        "distance": round(near, 2), "views": o["views"], "mask": f"masks/{oid}.png"})
        colour = np.array([255, 140, 0]) if movable else np.array([0, 200, 255])   # movable orange, articulated cyan
        if interactable:
            preview[mask] = 0.55 * pano[mask] + 0.45 * colour       # tinted, with a 5 px edge (cosmetic, untuned)
            edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
            preview[edge] = colour
    (out / "objects.json").write_text(json.dumps(records, indent=1))

    # Preview with labels (half resolution).
    preview = cv2.cvtColor(cv2.resize(preview, (W // 2, H // 2)), cv2.COLOR_RGB2BGR)
    for r in (r for r in records if r["interactable"]):
        ys, xs = np.nonzero(cv2.imread(str(out / r["mask"]), cv2.IMREAD_GRAYSCALE)[::2, ::2])
        if len(xs) and xs.max() - xs.min() < W // 4:     # over half the preview wide (as across the seam): no label
            text = r["label"]
            org = (int(xs.mean()) - 4 * len(text), int(ys.mean()))
            cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "labels.jpg"), preview)
    n, a = sum(r["movable"] for r in records), sum(r["kind"] == "articulated" for r in records)
    log(f"{scene.name}: {len(records)} objects, {n} movable, {a} articulated -> {out}")
    for r in records:
        log(f"  {'*' if r['interactable'] else ' '} {r['id']:28s} {r['kind']:11s} {r['distance']:5.2f} m  "
            f"size {' x '.join(f'{s:.2f}' for s in r['size'])}")
    return records


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("--max_size", type=float, default=1.5, help="m, largest side of a movable object (untuned; "
                                                               "leaves chairs movable)")
    p.add_argument("--min_thickness", type=float, default=0.01, help="m, smallest side of a movable object (untuned)")
    p.add_argument("--view_size", type=int, default=1024, help="px, each perspective view (untuned)")
    p.add_argument("--batch", type=int, default=17, help="requests sent to Triton at once "
                                                          "(17: every view in one round)")
    p.add_argument("--redetect", action="store_true", help="ignore objects/detections.json from an earlier run")
    a = p.parse_args()
    run(a.scene, models.Triton(), a.max_size, a.min_thickness, a.view_size, a.batch, a.redetect)
