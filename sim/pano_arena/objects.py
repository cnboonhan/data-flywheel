# /// script
# requires-python = ">=3.10"
# dependencies = ["transformers>=4.57", "accelerate", "torch", "torchvision", "opencv-python-headless", "pillow"]
# ///
"""Step 2: find the objects in a panorama and decide which are pickable.

    uv run sim/pano_arena/objects.py datasets/pano_arena/<scene>

Needs step 1 (depth/). The panorama is cut into perspective views (8 headings at the horizon and 45 deg down, one
straight down). Qwen3-VL names the kinds of object in each view and whether each is pickable (one hand can pick it up);
Grounding DINO boxes every instance of those names, and SAM 2.1 turns each box into a mask. Masks are mapped back onto
the panorama, and detections of the same object from overlapping views are merged. Each object gets a 3D box from the
depth. Pickable also needs at most --max_size across and at least --min_thickness thick; everything else stays in the
background.
Writes <scene>/objects/: objects.json, masks/<id>.png (panorama-sized), views/ (the perspective views and their
cameras, reused by later steps), labels.jpg (preview: pickable objects in colour, the rest grey).
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import common

PROMPT = (
    "List the kinds of physical objects in this indoor photo: furniture, appliances, devices, containers and small "
    "items on surfaces. Skip walls, floor, ceiling, windows, curtains and ceiling lights. Give each kind a short "
    "singular name (e.g. \"glass jar\") and say whether it is 'pickable' (one hand can pick it up and carry it: cups, "
    "bottles, jars, bowls, boxes, books, kettles, toasters) or 'fixed' (anything else: furniture, chairs, large "
    "appliances, built-in fittings). Answer only with a JSON object mapping each name to its kind.")

p = argparse.ArgumentParser()
p.add_argument("scene", type=Path)
p.add_argument("--max_size", type=float, default=0.6, help="m, largest side of a pickable object")
p.add_argument("--min_thickness", type=float, default=0.03, help="m, smallest side of a pickable object")
p.add_argument("--view_size", type=int, default=1024)
p.add_argument("--redetect", action="store_true", help="ignore objects/detections.json from an earlier run")
p.add_argument("--vlm", default="Qwen/Qwen3-VL-8B-Instruct")
p.add_argument("--detector", default="IDEA-Research/grounding-dino-base")
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

# Detect in two passes: Qwen3-VL names the kinds of object in each view (a short answer; asked for boxes, it starts
# repeating itself on rows of similar items), then Grounding DINO finds every instance of those names.
from torchvision.ops import nms
from transformers import (AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                          Sam2Processor)

cache = out / "detections.json"
if cache.exists() and not args.redetect:
    dets = json.loads(cache.read_text())
else:
    proc = AutoProcessor.from_pretrained(args.vlm)
    vlm = AutoModelForImageTextToText.from_pretrained(args.vlm, dtype=torch.bfloat16, device_map="cuda")
    names = []
    for img in images:
        msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(img)}, {"type": "text", "text": PROMPT}]}]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                     return_tensors="pt").to("cuda")
        with torch.no_grad():
            y = vlm.generate(**x, max_new_tokens=400, do_sample=False)
        text = proc.decode(y[0, x["input_ids"].shape[1]:], skip_special_tokens=True)
        pairs = re.findall(r'"([^"]+)"\s*:\s*"(pickable|fixed)"', text)
        names.append({n.lower().strip(): k for n, k in pairs})
    del vlm
    torch.cuda.empty_cache()

    gp = AutoProcessor.from_pretrained(args.detector)
    gd = AutoModelForZeroShotObjectDetection.from_pretrained(args.detector).cuda().eval()
    dets = []
    for i, (img, kinds) in enumerate(zip(images, names)):
        if not kinds:
            continue
        x = gp(images=Image.fromarray(img), text=". ".join(kinds) + ".", return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = gd(**x)
        r = gp.post_process_grounded_object_detection(o, x.input_ids, threshold=0.3, text_threshold=0.25,
                                                      target_sizes=[img.shape[:2]])[0]
        keep = nms(r["boxes"], r["scores"], 0.7).tolist()                 # one box per object across similar names
        for j in keep:
            t = r["text_labels"][j]
            label = next((n for n in kinds if n == t), None) or next((n for n in kinds if t and (t in n or n in t)), None)
            if label:
                dets.append({"view": i, "box": r["boxes"][j].tolist(), "label": label, "kind": kinds[label]})
        print(f"view {i:02d} (yaw {cams[i][0]}, pitch {cams[i][1]}): {len(kinds)} kinds, "
              f"{sum(d['view'] == i for d in dets)} objects", flush=True)
    del gd
    torch.cuda.empty_cache()
    cache.write_text(json.dumps(dets))

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

# Merge detections of one object from overlapping views (most of the smaller mask inside the larger). Views can
# disagree on the kind; the majority wins, ties going to pickable.
objs = []
for d in sorted((d for d in dets if len(d.get("pixels", ())) > 50), key=lambda d: -len(d["pixels"])):
    for o in objs:
        inter = len(np.intersect1d(o["pixels"], d["pixels"], assume_unique=True))
        if inter > 0.6 * min(len(o["pixels"]), len(d["pixels"])):
            o["pixels"] = np.union1d(o["pixels"], d["pixels"])
            o["labels"].append(d["label"])
            o["kinds"].append(d["kind"])
            o["views"].append({"view": d["view"], "box": d["box"]})
            break
    else:
        objs.append({"pixels": d["pixels"], "labels": [d["label"]], "kinds": [d["kind"]],
                     "views": [{"view": d["view"], "box": d["box"]}]})
for o in objs:
    o["kind"] = max(set(o["kinds"]), key=lambda k: (o["kinds"].count(k), ["fixed", "pickable"].index(k)))

# 3D extent from the depth: drop mask pixels that bled onto the background (far from the object's median distance).
records, preview = [], (pano * 0.35).astype(np.uint8)
gray = cv2.cvtColor(cv2.cvtColor(pano, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
for k, o in enumerate(objs):
    mask = np.zeros(H * W, bool)
    mask[o["pixels"]] = True
    mask = cv2.morphologyEx(mask.reshape(H, W).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    dist = distance[mask]
    keep = np.abs(dist - np.median(dist)) < 0.25 * np.median(dist) + 0.05
    pts = points[mask][keep]
    lo, hi = np.percentile(pts, 2, 0), np.percentile(pts, 98, 0)
    size, near = hi - lo, float(np.percentile(np.linalg.norm(pts[:, :2], axis=1), 5))
    label = Counter(o["labels"]).most_common(1)[0][0]
    pickable = o["kind"] == "pickable" and args.min_thickness <= size.min() and size.max() <= args.max_size
    oid = f"{k:02d}_{re.sub(r'[^a-z0-9]+', '_', label).strip('_')}"
    cv2.imwrite(str(out / "masks" / f"{oid}.png"), mask.astype(np.uint8) * 255)
    records.append({"id": oid, "label": label, "kind": o["kind"], "pickable": bool(pickable),
                    "center": ((lo + hi) / 2).round(3).tolist(), "size": size.round(3).tolist(),
                    "distance": round(near, 2), "views": o["views"], "mask": f"masks/{oid}.png"})
    colour = np.array(cv2.applyColorMap(np.uint8([[k * 47 % 255]]), cv2.COLORMAP_TURBO)[0, 0][::-1])
    preview[mask] = (0.5 * pano[mask] + 0.5 * colour) if pickable else gray[mask] * 0.7
(out / "objects.json").write_text(json.dumps(records, indent=1))

# Preview with labels (half resolution).
preview = cv2.cvtColor(cv2.resize(preview, (W // 2, H // 2)), cv2.COLOR_RGB2BGR)
for r in (r for r in records if r["pickable"]):
    ys, xs = np.nonzero(cv2.imread(str(out / r["mask"]), cv2.IMREAD_GRAYSCALE)[::2, ::2])
    if len(xs) and xs.max() - xs.min() < W // 4:
        text = f"{r['label']} {r['distance']:.1f}m"
        org = (int(xs.mean()) - 4 * len(text), int(ys.mean()))
        cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(preview, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(out / "labels.jpg"), preview)
n = sum(r["pickable"] for r in records)
print(f"{args.scene.name}: {len(records)} objects, {n} pickable -> {out}")
for r in records:
    print(f"  {'*' if r['pickable'] else ' '} {r['id']:28s} {r['kind']:8s} {r['distance']:5.2f} m  "
          f"size {' x '.join(f'{s:.2f}' for s in r['size'])}")
