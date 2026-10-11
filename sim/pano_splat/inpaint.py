# /// script
# requires-python = ">=3.10"
# dependencies = ["moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebe", "diffusers>=0.36", "transformers", "accelerate", "bitsandbytes", "peft"]
# [tool.uv]
# override-dependencies = ["moderngl; sys_platform == 'never'"]   # MoGe's renderer, unused here; no aarch64 wheel
# ///
"""Step 3: erase the movable objects from the panorama.

    uv run sim/pano_splat/inpaint.py datasets/pano_splat/<scene>

Needs steps 1 and 2. A perspective view around each movable object is inpainted and pasted back onto the panorama:
LaMa for small holes; each connected part of a hole covering more than --large of the view goes, one at a time, to
Qwen-Image-Edit-2511 with an object-removal LoRA and the 8-step Lightning LoRA (all Apache-2.0), the part tinted red
(LaMa smears big holes; SDXL inpainting paints new objects into them; the LoRA leaves several marked objects in place).
Qwen's 20B transformer must stay bf16 (4-bit NF4 turns its output to grain). On a GPU of 48 GB or more everything runs
in bf16 on the GPU, 40 steps with guidance; under 48 GB the transformer is streamed from CPU memory (~40 GB of RAM), the
text encoder is 4-bit and the 8-step Lightning LoRA replaces guidance. The depth behind the objects comes from MoGe on the cleaned
panorama, scaled to match the original depth around each hole.
Writes <scene>/inpaint/: pano.jpg, distance.npy, mask.png (what was erased), before_after.jpg.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import common

LAMA = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"   # TorchScript

p = argparse.ArgumentParser()
p.add_argument("scene", type=Path)
p.add_argument("--size", type=int, default=1024, help="crop size in pixels")
p.add_argument("--grow", type=int, default=12, help="px to grow masks by at 4096 wide (edges, contact shadows)")
p.add_argument("--large", type=float, default=0.03, help="view fraction of a hole part above which Qwen-Image-Edit inpaints")
p.add_argument("--editor", default="Qwen/Qwen-Image-Edit-2511")
p.add_argument("--remover", default="prithivMLmods/QIE-2511-Object-Remover-v2", help="object-removal LoRA")
p.add_argument("--lightning", default="lightx2v/Qwen-Image-Edit-2511-Lightning")
p.add_argument("--model", default="Ruicheng/moge-3-vitg")
p.add_argument("--fill", choices=["edit", "mask", "regen"], default="edit",
               help="large holes: edit = tint the object red and ask the removal LoRA to remove it; mask = LaMa first, "
                    "then Qwen-Image-Edit inpainting regenerates exactly the hole; regen = Qwen-Image-Edit redraws the "
                    "crop without the objects, named in the prompt, and the hole is taken from it (mask and regen need "
                    "48 GB of GPU memory)")
p.add_argument("--strength", type=float, default=1.0, help="--fill mask: how much of the LaMa start is regenerated")
p.add_argument("--shadow", type=float, default=0.0,
               help="m: also erase the surface the object stands on (floor, tabletop) this far around its footprint")
p.add_argument("--out", default="inpaint", help="output folder in the scene")
args = p.parse_args()
out = args.scene / args.out
out.mkdir(exist_ok=True)

pano = cv2.cvtColor(cv2.imread(str(args.scene / "depth" / "pano.jpg")), cv2.COLOR_BGR2RGB)
distance = np.load(args.scene / "depth" / "distance.npy")
frame = common.load_frame(args.scene)
objects = [o for o in json.loads((args.scene / "objects" / "objects.json").read_text()) if o["movable"]]
H, W = distance.shape
R = np.array(frame["R_moge_to_level"])
dirs = common.directions(H, W)
grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (args.grow * W // 4096) + 1,) * 2)
masks = {o["id"]: cv2.imread(str(args.scene / "objects" / o["mask"]), cv2.IMREAD_GRAYSCALE) > 0 for o in objects}

if args.shadow > 0:   # contact shadows: pixels on the supporting surface (at the object's base height) near its footprint
    pts, cell = common.points(distance, frame).astype(np.float32), 0.02
    r = int(round(args.shadow / cell))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    for oid, m in masks.items():
        ys, xs = np.nonzero(m)
        d = distance[ys, xs]
        keep = np.abs(d - np.median(d)) < 0.25 * np.median(d) + 0.05     # not the background the mask bled onto
        p = pts[ys[keep], xs[keep]]
        base, lo = np.percentile(p[:, 2], 2), p[:, :2].min(0) - (r + 1) * cell
        g = ((p[:, :2] - lo) / cell).astype(int)
        grid = np.zeros(g.max(0) + r + 3, np.uint8)
        grid[g[:, 0], g[:, 1]] = 1
        grid = cv2.dilate(grid, disk)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1          # search the mask's box, grown by its own size
        y0, y1 = max(ys.min() - h, 0), min(ys.max() + h + 1, H)
        x0, x1 = max(xs.min() - w, 0), min(xs.max() + w + 1, W)
        sub = pts[y0:y1, x0:x1]
        gi = ((sub[..., :2] - lo) / cell).astype(int)
        inside = (gi >= 0).all(-1) & (gi < grid.shape).all(-1)
        hit = np.zeros(inside.shape, bool)
        hit[inside] = grid[gi[inside][:, 0], gi[inside][:, 1]] > 0
        m[y0:y1, x0:x1] |= hit & (np.abs(sub[..., 2] - base) < 0.03)


def crop_camera(mask):
    """A view centred on the mask: yaw/pitch of its mean direction (scene frame), fov 1.5x its angular extent."""
    d = dirs[mask][:: max(1, mask.sum() // 20000)]
    c = (d.mean(0) @ R.T)
    c /= np.linalg.norm(c)
    extent = np.degrees(np.percentile(np.arccos(np.clip(d @ R.T @ c, -1, 1)), 98))
    yaw, pitch = np.degrees(np.arctan2(c[1], c[0])), np.degrees(np.arcsin(c[2]))
    return {"yaw": round(float(yaw), 3), "pitch": round(float(pitch), 3),
            "fov": round(float(np.clip(3 * extent + 10, 40, 120)), 3), "size": args.size}


# Erase: per object (largest first), LaMa on its crop with every still-unfilled removed pixel masked; the central 90%
# of the crop is pasted back, and those pixels count as filled.
ckpt = Path(torch.hub.get_dir()) / "checkpoints" / "big-lama.pt"
if not ckpt.exists():
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(LAMA, str(ckpt))
lama = torch.jit.load(str(ckpt), map_location="cuda").eval()
qwen = qwen_mask = qwen_regen = None


def remove_large(img, hole):
    """Qwen-Image-Edit + the remover LoRA on a crop; the hole's colours are matched to the input on a ring around it."""
    global qwen
    from PIL import Image
    if qwen is None:
        from diffusers import PipelineQuantizationConfig, QwenImageEditPlusPipeline
        small = torch.cuda.get_device_properties(0).total_memory < 48e9
        q = PipelineQuantizationConfig(quant_backend="bitsandbytes_4bit", components_to_quantize=["text_encoder"],
                                       quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                                     "bnb_4bit_compute_dtype": torch.bfloat16}) if small else None
        qwen = QwenImageEditPlusPipeline.from_pretrained(args.editor, quantization_config=q, dtype=torch.bfloat16)
        if small:
            qwen.transformer.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                                  offload_type="block_level", num_blocks_per_group=4)
            qwen.text_encoder.to("cuda")
            qwen.vae.to("cuda")
        else:
            qwen.to("cuda")
        qwen.load_lora_weights(args.remover, weight_name="Qwen-Image-Edit-2511-Object-Remover-v2-9200.safetensors",
                               adapter_name="remove")
        if small:   # 8 steps without guidance instead of 40 with it
            qwen.load_lora_weights(args.lightning, weight_name="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
                                   adapter_name="lightning")
        qwen.set_adapters(["remove", "lightning"] if small else ["remove"], [1.0, 1.0] if small else [1.0])
        qwen.fast = small
        qwen.set_progress_bar_config(disable=True)
    m = hole[..., None].astype(np.float32)
    marked = (img * (1 - 0.6 * m) + np.array([255, 0, 0]) * 0.6 * m).astype(np.uint8)   # a red box removes less
    res = qwen(image=[Image.fromarray(marked)], prompt="Remove the red highlighted object from the scene.",
               negative_prompt=" ", true_cfg_scale=1.0 if qwen.fast else 4.0, num_inference_steps=8 if qwen.fast else 40,
               generator=torch.Generator("cuda").manual_seed(0)).images[0]
    res = np.array(res.resize(img.shape[1::-1], Image.LANCZOS)).astype(np.float32)
    ring = (cv2.dilate(hole, np.ones((25, 25), np.uint8)) > 0) & (hole == 0)
    res += (img[ring].astype(np.float32) - res[ring]).mean(0)
    return np.where(m > 0, res.clip(0, 255), img).astype(np.uint8)


def fill_mask(img, hole):
    """Qwen-Image-Edit inpainting of exactly the hole, starting from the LaMa fill in img."""
    global qwen_mask
    from PIL import Image
    if qwen_mask is None:
        from diffusers import QwenImageEditInpaintPipeline
        qwen_mask = QwenImageEditInpaintPipeline.from_pretrained(args.editor, dtype=torch.bfloat16).to("cuda")
        qwen_mask.set_progress_bar_config(disable=True)
    res = qwen_mask(image=Image.fromarray(img), mask_image=Image.fromarray(hole * 255),
                    prompt="Fill the masked area with the background around it: continue the floor, walls, furniture "
                           "surfaces and lighting naturally. Do not add any objects.",
                    negative_prompt="object, clutter, chair, bag, box, bottle, person, text, blur",
                    true_cfg_scale=4.0, strength=args.strength, num_inference_steps=40, height=img.shape[0],
                    width=img.shape[1], generator=torch.Generator("cuda").manual_seed(0)).images[0]
    res = np.array(res.resize(img.shape[1::-1], Image.LANCZOS))
    return np.where(hole[..., None] > 0, res, img).astype(np.uint8)


def fill_regen(img, hole, labels):
    """The crop redrawn by Qwen-Image-Edit without the named objects; None if the hole barely changed twice (the model
    left an object in place)."""
    global qwen_regen
    from PIL import Image
    if qwen_regen is None:
        from diffusers import QwenImageEditPlusPipeline
        qwen_regen = QwenImageEditPlusPipeline.from_pretrained(args.editor, dtype=torch.bfloat16).to("cuda")
        qwen_regen.set_progress_bar_config(disable=True)
    names = ", ".join(f"the {n}" for n in dict.fromkeys(labels))
    m = hole > 0
    ring = (cv2.dilate(hole, np.ones((25, 25), np.uint8)) > 0) & ~m
    for seed in (0, 1):
        res = qwen_regen(image=[Image.fromarray(img)], prompt=f"Remove {names} from the scene, together with their "
                         "shadows. Keep everything else exactly the same.", negative_prompt=" ", true_cfg_scale=4.0,
                         num_inference_steps=40, generator=torch.Generator("cuda").manual_seed(seed)).images[0]
        res = np.array(res.resize(img.shape[1::-1], Image.LANCZOS)).astype(np.float32)
        res += (img[ring].astype(np.float32) - res[ring]).mean(0)     # match the colours around the hole
        if np.abs(res[m] - img[m]).mean() > 25:
            return np.where(m[..., None], res.clip(0, 255), img).astype(np.uint8)
    return None


removed = cv2.dilate(np.any(list(masks.values()), 0).astype(np.uint8), grow) > 0
todo, clean, n_large = removed.copy(), pano.copy(), 0
for o in sorted(objects, key=lambda o: -masks[o["id"]].sum()):
    cam = crop_camera(masks[o["id"]])
    mx, my = common.view(frame, H, W, cam["yaw"], cam["pitch"], cam["size"], cam["fov"])
    hole = cv2.remap(todo.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    if not hole.any():
        continue
    img = cv2.remap(clean, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    # Each connected part of the hole separately: the removal LoRA leaves several highlighted objects in place.
    n, part, stats, _ = cv2.connectedComponentsWithStats(hole)
    big = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > args.large * hole.size]
    small = (hole > 0) if args.fill != "edit" else (hole > 0) & ~np.isin(part, big)   # LaMa: the start or fallback
    fill = img
    if small.any():
        with torch.no_grad():
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float().cuda() / 255
            m = torch.from_numpy(small)[None, None].float().cuda()
            lamafill = (lama(x, m)[0].permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        fill = np.where(small[..., None], lamafill[:cam["size"], :cam["size"]], img)
    if args.fill == "regen" and big:
        labels = [p["label"] for p in objects if (cv2.remap(masks[p["id"]].astype(np.uint8), mx, my, cv2.INTER_NEAREST,
                                                             borderMode=cv2.BORDER_WRAP) & hole).any()]
        regen = fill_regen(img, hole, labels)
        print(f"  {o['id']}: Qwen-Image-Edit redraw ({hole.mean() * 100:.0f}% of the view)"
              + ("" if regen is not None else ": object left in place, inpainting instead"), flush=True)
        fill, n_large, big = (regen if regen is not None else fill_mask(fill, hole)), n_large + 1, []
    if args.fill == "mask" and big:
        fill, n_large = fill_mask(fill, (hole > 0).astype(np.uint8)), n_large + 1
        print(f"  {o['id']}: Qwen-Image-Edit inpaint ({hole.mean() * 100:.0f}% of the view)", flush=True)
        big = []
    for k in big:
        fill, n_large = remove_large(fill, (part == k).astype(np.uint8)), n_large + 1
        print(f"  {o['id']}: Qwen-Image-Edit ({stats[k, cv2.CC_STAT_AREA] / hole.size * 100:.0f}% of the view)", flush=True)
    idx = np.flatnonzero(todo)
    px, py, front = common.project(frame, dirs.reshape(-1, 3)[idx], cam["yaw"], cam["pitch"], cam["size"], cam["fov"])
    lo, hi = 0.05 * cam["size"], 0.95 * cam["size"]
    inside = front & (px > lo) & (px < hi) & (py > lo) & (py < hi)
    idx, px, py = idx[inside], px[inside], py[inside]
    if not len(idx):                          # the hole lies at the crop's edge; a later crop or Telea fills it
        continue
    pad = -len(idx) % 4096                    # remap takes at most 32767 rows: sample through a 4096-wide grid
    grid = [np.pad(a, (0, pad)).reshape(-1, 4096) for a in (px, py)]
    clean.reshape(-1, 3)[idx] = cv2.remap(fill, *grid, cv2.INTER_LINEAR).reshape(-1, 3)[:len(idx)]
    todo.reshape(-1)[idx] = False
left = int(todo.sum())
if left:                                         # pixels no crop covered well (huge objects): classical inpainting
    clean = cv2.inpaint(clean, todo.astype(np.uint8), 5, cv2.INPAINT_TELEA)
del lama, qwen, qwen_mask, qwen_regen
torch.cuda.empty_cache()

# Depth behind the removed objects: MoGe on the clean panorama, scaled per hole to the original depth around it.
new, _ = common.moge_distance(clean, args.model)
filled = distance.copy()
n, labels, stats, _ = cv2.connectedComponentsWithStats(removed.astype(np.uint8))
ring_k = np.ones((41, 41), np.uint8)
for k in range(1, n):
    x0, y0, w, h = stats[k, :4]
    x0, y0, x1, y1 = max(x0 - 30, 0), max(y0 - 30, 0), min(x0 + w + 30, W), min(y0 + h + 30, H)
    hole = labels[y0:y1, x0:x1] == k
    ring = (cv2.dilate(hole.astype(np.uint8), ring_k) > 0) & ~removed[y0:y1, x0:x1]
    scale = np.median(distance[y0:y1, x0:x1][ring] / new[y0:y1, x0:x1][ring]) if ring.any() else 1.0
    filled[y0:y1, x0:x1][hole] = new[y0:y1, x0:x1][hole] * scale

np.save(out / "distance.npy", filled.astype(np.float32))
cv2.imwrite(str(out / "pano.jpg"), cv2.cvtColor(clean, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(str(out / "mask.png"), removed.astype(np.uint8) * 255)
half = lambda a: cv2.resize(a, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(out / "before_after.jpg"), cv2.cvtColor(np.vstack([half(pano), half(clean)]), cv2.COLOR_RGB2BGR))
print(f"{args.scene.name}: erased {len(objects)} objects ({removed.mean() * 100:.1f}% of the panorama, "
      f"{n_large} holes with Qwen-Image-Edit"
      f"{f', {left} px by classical inpainting' if left else ''}) -> {out}")
