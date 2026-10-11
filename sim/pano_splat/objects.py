# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers>=4.57", "accelerate", "torch", "torchvision", "opencv-python-headless", "pillow"]
# ///
"""Step 2: find the objects in a panorama and which are interactable: movable or articulated.

    uv run sim/pano_splat/objects.py datasets/pano_splat/<scene>

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
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import common

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

p = argparse.ArgumentParser()
p.add_argument("scene", type=Path)
p.add_argument("--max_size", type=float, default=1.5, help="m, largest side of a movable object")
p.add_argument("--min_thickness", type=float, default=0.01, help="m, smallest side of a movable object")
p.add_argument("--view_size", type=int, default=1024)
p.add_argument("--batch", type=int, default=17, help="VLM answers asked or generated together")
p.add_argument("--redetect", action="store_true", help="ignore objects/detections.json from an earlier run")
p.add_argument("--vlm", default="Qwen/Qwen3.5-122B-A10B", help="on Triton if TRITON_URL is set (slurm.env)")
p.add_argument("--local_vlm", action="store_true", help="load the VLM here instead (bf16, ~245 GB: one B300)")
p.add_argument("--detector", default="iSEE-Laboratory/llmdet_large", help="open-vocabulary detector for recall")
p.add_argument("--sam", default="facebook/sam2.1-hiera-large")
args = p.parse_args()
src, out = args.scene / "depth", args.scene / "objects"
shutil.rmtree(out / "masks", ignore_errors=True)
(out / "masks").mkdir(parents=True)
(out / "views").mkdir(exist_ok=True)

pano = cv2.cvtColor(cv2.imread(str(src / "pano.jpg")), cv2.COLOR_BGR2RGB)
distance = np.load(src / "distance.npy")
frame = common.load_frame(args.scene)
H, W = distance.shape

points = common.points(distance, frame)


cams = [(yaw, pitch) for pitch in (0, -45) for yaw in range(0, 360, 45)] + [(0, -90)]
grids, images = [], []
for i, (yaw, pitch) in enumerate(cams):
    mx, my = common.view(frame, H, W, yaw, pitch, args.view_size)
    grids.append((mx, my))
    images.append(cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
    cv2.imwrite(str(out / "views" / f"{i:02d}.jpg"), cv2.cvtColor(images[-1], cv2.COLOR_RGB2BGR))
(out / "views" / "cameras.json").write_text(json.dumps(
    [{"view": i, "yaw": y, "pitch": pt, "fov": 90.0, "size": args.view_size} for i, (y, pt) in enumerate(cams)]))

# Detect: the VLM lists the objects in each view with their kind and box; LLMDet then looks for every name found
# anywhere in the scene, in every view, and adds the boxes the VLM missed.
from collections import defaultdict

from torchvision.ops import box_iou, nms
from transformers import (AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                          Sam2Processor)

vlm = proc = None


def ask(imgs, texts, max_new_tokens):
    """The VLM's answers, one per (image, question): from the VLM on Triton when TRITON_URL is set (slurm.env), --batch
    requests at a time; otherwise generated here in batches (decoding is memory-bound, so a batch takes about as long
    as one answer), the model loaded on first use and kept: detection and the check share it."""
    global vlm, proc
    if os.environ.get("TRITON_URL") and not args.local_vlm:
        from concurrent.futures import ThreadPoolExecutor
        if proc is None:
            proc = AutoProcessor.from_pretrained(args.vlm)
        with ThreadPoolExecutor(args.batch) as pool:
            return list(pool.map(lambda it: common.triton_ask(proc, args.vlm, it[1], it[0], max_new_tokens),
                                 zip(imgs, texts)))
    if vlm is None:
        proc = AutoProcessor.from_pretrained(args.vlm)
        proc.tokenizer.padding_side = "left"
        vlm = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype=torch.bfloat16, device_map="auto")
    answers = []
    for i in range(0, len(imgs), args.batch):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)},
                                              {"type": "text", "text": text}]}]
                for img, text in zip(imgs[i:i + args.batch], texts[i:i + args.batch])]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, padding=True,
                                     return_tensors="pt", enable_thinking=False).to(vlm.device)
        with torch.no_grad():
            y = vlm.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        answers += proc.batch_decode(y[:, x["input_ids"].shape[1]:], skip_special_tokens=True)
    return answers


cache = out / "detections.json"
if cache.exists() and not args.redetect:
    dets = json.loads(cache.read_text())
else:
    dets, kinds = [], defaultdict(Counter)
    for i, answer in enumerate(ask(images, [PROMPT] * len(images), 3000)):
        seen = set()
        for m in re.finditer(r"\{[^{}]*\}", answer):
            try:
                d = json.loads(m.group(0))
            except ValueError:
                continue
            label, kind, b = str(d.get("label", "")).lower().strip(), d.get("kind"), d.get("bbox_2d")
            if not label or kind not in KINDS or not isinstance(b, list) or len(b) != 4:
                continue
            box = [float(c) * args.view_size / 1000 for c in b]
            key = (label, *(round(c / 8) for c in box))               # the VLM sometimes repeats itself
            if key in seen or box[2] - box[0] < 4 or box[3] - box[1] < 4:
                continue
            seen.add(key)
            dets.append({"view": i, "box": box, "label": label, "kind": kind, "source": "vlm"})
            kinds[label][kind] += 1
        print(f"view {i:02d} (yaw {cams[i][0]}, pitch {cams[i][1]}): {len(seen)} objects", flush=True)

    labels = sorted(kinds)
    kind_of = {n: max(c, key=lambda k: (c[k], KINDS.index(k))) for n, c in kinds.items()}
    gp = AutoProcessor.from_pretrained(args.detector)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).cuda().eval()
    added = 0
    for i, img in enumerate(images):
        boxes, scores, names = [], [], []
        for j in range(0, len(labels), 20):                          # the text prompt holds ~256 tokens
            chunk = labels[j:j + 20]
            x = gp(images=Image.fromarray(img), text=". ".join(chunk) + ".", return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = gd(**x)
            r = gp.post_process_grounded_object_detection(o, x.input_ids, threshold=0.35, text_threshold=0.25,
                                                          target_sizes=[img.shape[:2]])[0]
            for b, sc, t in zip(r["boxes"], r["scores"], r["text_labels"]):
                n = next((n for n in chunk if n == t), None) or next((n for n in chunk if t and (t in n or n in t)), None)
                if n:
                    boxes.append(b.cpu()), scores.append(sc.cpu()), names.append(n)
        if not boxes:
            continue
        B, S = torch.stack(boxes), torch.stack(scores)
        mine = torch.tensor([d["box"] for d in dets if d["view"] == i]).reshape(-1, 4)
        for k in nms(B, S, 0.7).tolist():
            if len(mine) and box_iou(B[k:k + 1], mine).max() > 0.5:
                continue
            dets.append({"view": i, "box": B[k].tolist(), "label": names[k], "kind": kind_of[names[k]], "source": "llmdet"})
            added += 1
    print(f"{sum(d['source'] == 'vlm' for d in dets)} boxes from the VLM, {added} more from LLMDet", flush=True)
    del gd
    torch.cuda.empty_cache()
    cache.write_text(json.dumps(dets))

# The VLM sometimes loops, listing one label in a row of equal boxes stepping across the view: drop such runs.
runs = Counter((d["view"], d["label"], round(d["box"][1] / 8), round(d["box"][3] / 8),
                round((d["box"][2] - d["box"][0]) / 8)) for d in dets if d.get("source", "vlm") == "vlm")
dets = [d for d in dets if d.get("source", "vlm") != "vlm" or runs[(d["view"], d["label"], round(d["box"][1] / 8),
        round(d["box"][3] / 8), round((d["box"][2] - d["box"][0]) / 8))] < 4]

# Segment each box with SAM 2.1 and map the mask onto the panorama.
sp = Sam2Processor.from_pretrained(args.sam)
sam = Sam2Model.from_pretrained(args.sam).cuda().eval()
for i, img in enumerate(images):
    mine = [d for d in dets if d["view"] == i]
    if not mine:
        continue
    x = sp(images=Image.fromarray(img), input_boxes=[[d["box"] for d in mine]], return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = sam(**x, multimask_output=False)
    masks = sp.post_process_masks(o.pred_masks.cpu(), x["original_sizes"])[0][:, 0].numpy()
    mx, my = grids[i]
    for d, m in zip(mine, masks):
        px = np.round(mx[m]).astype(int) % W, np.clip(np.round(my[m]).astype(int), 0, H - 1)
        d["pixels"] = np.unique(px[1] * W + px[0])
del sam
torch.cuda.empty_cache()

# Merge detections of one object from overlapping views and detectors (most of the smaller mask inside the larger,
# and the same label or a similar size, so a cat stays apart from the cat tree it sits on). Views can disagree on the kind; the majority wins, ties
# going to movable.
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
    """Mask, 3D box (scene frame) and nearest horizontal distance of an object, without the mask pixels that bled
    onto the background (far from the object's median distance)."""
    mask = np.zeros(H * W, bool)
    mask[pixels] = True
    mask = cv2.morphologyEx(mask.reshape(H, W).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    dist = distance[mask]
    keep = np.abs(dist - np.median(dist)) < 0.25 * np.median(dist) + 0.05
    pts = points[mask][keep]
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    return mask, lo, hi, float(np.percentile(np.linalg.norm(pts[:, :2], axis=1), 5))


# Second merge, in 3D: parts of one object that overlap too little in the panorama (a chair's base whose mask spilled
# onto the floor, a part cut at a view's edge) but share a label and most of their footprint on the floor plan.
for o in objs:
    o["lo"], o["hi"] = extent(o["pixels"])[1:3]
merged = True
while merged:
    merged = False
    for a in sorted(objs, key=lambda o: -len(o["pixels"])):
        for b in objs:
            if b is a or len(b["pixels"]) > len(a["pixels"]) or not set(a["labels"]) & set(b["labels"]):
                continue
            ov = np.clip(np.minimum(a["hi"][:2], b["hi"][:2]) - np.maximum(a["lo"][:2], b["lo"][:2]), 0, None).prod()
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

# Check each movable candidate on its own: the per-view naming misjudges objects seen out of context (a tabletop
# with items on it from above). The VLM sees the view where the object is largest, boxed in red. Cached per box.
checks_file = out / "checks.json"
checks = {} if args.redetect or not checks_file.exists() else json.loads(checks_file.read_text())
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
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 0), max(2, args.view_size // 256))
        todo.append((o, img))
texts = [CHECK.format(label=o["label"], movable=MOVABLE, articulated=ARTICULATED, static=STATIC) for o, _ in todo]
for (o, _), answer in zip(todo, ask([img for _, img in todo], texts, 10)):
    checks[o["key"]] = next((k for k in KINDS if k in answer.lower()), o["kind"])
    if checks[o["key"]] != o["kind"]:
        print(f"check: {o['label']} {o['kind']} -> {checks[o['key']]}", flush=True)
for o in objs:
    if o["kind"] != "static":
        o["kind"] = checks[o["key"]]
checks_file.write_text(json.dumps(checks, indent=1))
del vlm
torch.cuda.empty_cache()

records, preview = [], (pano * 0.45).astype(np.uint8)
for k, o in enumerate(objs):
    mask, lo, hi, near = extent(o["pixels"])
    size = hi - lo
    label = Counter(o["labels"]).most_common(1)[0][0]
    movable = o["kind"] == "movable" and args.min_thickness <= size.min() and size.max() <= args.max_size
    oid = f"{k:02d}_{re.sub(r'[^a-z0-9]+', '_', label).strip('_')}"
    cv2.imwrite(str(out / "masks" / f"{oid}.png"), mask.astype(np.uint8) * 255)
    interactable = movable or o["kind"] == "articulated"
    records.append({"id": oid, "label": label, "kind": o["kind"], "movable": bool(movable),
                    "interactable": bool(interactable),
                    "center": ((lo + hi) / 2).round(3).tolist(), "size": size.round(3).tolist(),
                    "distance": round(near, 2), "views": o["views"], "mask": f"masks/{oid}.png"})
    colour = np.array([255, 140, 0]) if movable else np.array([0, 200, 255])   # movable orange, articulated cyan
    if interactable:
        preview[mask] = 0.55 * pano[mask] + 0.45 * colour
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        preview[edge] = colour
(out / "objects.json").write_text(json.dumps(records, indent=1))

# Preview with labels (half resolution).
preview = cv2.cvtColor(cv2.resize(preview, (W // 2, H // 2)), cv2.COLOR_RGB2BGR)
for r in (r for r in records if r["interactable"]):
    ys, xs = np.nonzero(cv2.imread(str(out / r["mask"]), cv2.IMREAD_GRAYSCALE)[::2, ::2])
    if len(xs) and xs.max() - xs.min() < W // 4:
        text = r["label"]
        org = (int(xs.mean()) - 4 * len(text), int(ys.mean()))
        cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(out / "labels.jpg"), preview)
n, a = sum(r["movable"] for r in records), sum(r["kind"] == "articulated" for r in records)
print(f"{args.scene.name}: {len(records)} objects, {n} movable, {a} articulated -> {out}")
for r in records:
    print(f"  {'*' if r['interactable'] else ' '} {r['id']:28s} {r['kind']:11s} {r['distance']:5.2f} m  "
          f"size {' x '.join(f'{s:.2f}' for s in r['size'])}")
