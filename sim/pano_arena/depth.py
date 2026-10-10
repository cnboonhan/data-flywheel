# /// script
# requires-python = ">=3.10"
# dependencies = ["moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebe"]
# ///
"""Step 1: metric depth of an equirectangular panorama, and the frame the rest of pano_arena works in.

    uv run sim/pano_arena/depth.py datasets/pano_arena/<scene> [--camera_height 1.5]

Reads <scene>/source.jpg. MoGe-2 runs on 20 perspective views (MoGe's own panorama split); their depths are merged in
the gradient domain (MoGe's merge), which loses the absolute scale, so the merged map is rescaled to the views' metric
depths. The floor is the lowest large horizontal layer of points. The output frame is Arena's: z up, origin on the floor
below the camera, +x towards the middle of the panorama. --camera_height rescales to a known height above the floor.
Writes <scene>/depth/: distance.npy (metres from the camera, per panorama pixel), pano.jpg (the image at that size),
points.ply (coloured points in the output frame), depth.png and floor.png (previews), frame.json.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import utils3d_moge as utils3d
from moge.model.v2 import MoGeModel
from moge.utils.panorama import (get_panorama_cameras, merge_panorama_depth, spherical_uv_to_directions,
                                 split_panorama_image)
from moge.utils.vis import colorize_depth

p = argparse.ArgumentParser()
p.add_argument("scene", type=Path)
p.add_argument("--camera_height", type=float, help="metres above the floor; default: as estimated")
p.add_argument("--width", type=int, default=4096, help="panorama width to work at")
p.add_argument("--model", default="Ruicheng/moge-2-vitl-normal")
args = p.parse_args()
out = args.scene / "depth"
out.mkdir(exist_ok=True)

image = cv2.cvtColor(cv2.imread(str(args.scene / "source.jpg")), cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (args.width, args.width // 2), interpolation=cv2.INTER_AREA)
h, w = image.shape[:2]

# Per-view metric depth.
extr, intr = get_panorama_cameras()
views = split_panorama_image(image, extr, intr, 512)
model = MoGeModel.from_pretrained(args.model).cuda().eval()
dist, masks = [], []
for i in range(0, len(views), 4):
    x = torch.tensor(np.stack(views[i:i + 4]) / 255, dtype=torch.float32, device="cuda").permute(0, 3, 1, 2)
    fov_x = torch.tensor(np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(intr[i:i + 4])))[0], device="cuda")
    with torch.no_grad():
        o = model.infer(x, fov_x=fov_x, apply_mask=False)
    dist += list(o["points"].norm(dim=-1).cpu().numpy())
    masks += list(o["mask"].cpu().numpy())
del model
torch.cuda.empty_cache()

# Merge, then restore the scale: median ratio of each view's metric distance to the merged one where they overlap.
mw, mh = min(1920, w), min(960, h)
merged, valid = merge_panorama_depth(mw, mh, dist, masks, extr, intr)
dirs = spherical_uv_to_directions(utils3d.np.uv_map(mh, mw))
ratios = []
for d, m, e, k in zip(dist, masks, extr, intr):
    uv, z = utils3d.np.project_cv(dirs, extrinsics=e, intrinsics=k)
    ok = (z > 0) & (uv > 0).all(-1) & (uv < 1).all(-1) & valid
    px = utils3d.np.uv_to_pixel(uv[ok], d.shape).astype(int)
    ok_px = m[px[:, 1], px[:, 0]]
    ratios.append(d[px[ok_px, 1], px[ok_px, 0]] / merged[ok][ok_px])
scale = float(np.median(np.concatenate(ratios)))
merged *= scale
distance = cv2.resize(merged, (w, h), interpolation=cv2.INTER_LINEAR)
valid = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

# MoGe's panorama frame: z up, the middle column (u = 0.5) looks along -x. Turn it 180 deg about z so it looks along +x.
pts = distance[..., None] * spherical_uv_to_directions(utils3d.np.uv_map(h, w)) * np.array([-1, -1, 1])

# Floor: the lowest z layer (2 cm bins) holding at least 2% of the points below the camera, refined with a plane fit.
below = pts[valid & (pts[..., 2] < -0.3)]
hist, edges = np.histogram(below[:, 2], bins=np.arange(below[:, 2].min(), -0.3, 0.02))
layer = np.flatnonzero(hist > 0.02 * len(below))[0]
z0 = edges[layer:layer + 2].mean()
layer_pts = below[np.abs(below[:, 2] - z0) < 0.05]
c = layer_pts.mean(0)
n = np.linalg.svd(layer_pts - c, full_matrices=False)[2][2]
n = n if n[2] > 0 else -n
estimated = float(-n @ c)                                # camera's height above the floor plane

# Level the floor (rotate n onto +z), put the origin under the camera, rescale to --camera_height if given.
axis = np.cross(n, [0, 0, 1])
R = cv2.Rodrigues(axis / (np.linalg.norm(axis) + 1e-12) * np.arcsin(min(np.linalg.norm(axis), 1.0)))[0]
s = args.camera_height / estimated if args.camera_height else 1.0
pts = (pts @ R.T) * s
pts[..., 2] += estimated * s
distance *= s

np.save(out / "distance.npy", distance.astype(np.float32))
cv2.imwrite(str(out / "pano.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
cv2.imwrite(str(out / "depth.png"), cv2.cvtColor(colorize_depth(distance, mask=valid), cv2.COLOR_RGB2BGR))
step = max(1, w // 1024)
sel = valid[::step, ::step]
trimesh.PointCloud(pts[::step, ::step][sel], image[::step, ::step][sel]).export(out / "points.ply")

# Top-down preview: points up to 2 m high, 1 cm per pixel, coloured by height; the camera is the red dot.
top = pts[valid & (pts[..., 2] < 2.0) & (np.abs(pts[..., :2]) < 8).all(-1)]
res, half = 0.01, 8.0
img = np.full((int(2 * half / res),) * 2 + (3,), 255, np.uint8)
order = np.argsort(top[:, 2])
ij = ((half - top[order][:, [0, 1]]) / res).astype(int)
img[ij[:, 0], ij[:, 1]] = cv2.applyColorMap((np.clip(top[order][:, 2] / 2.0, 0, 1) * 255).astype(np.uint8)[:, None],
                                            cv2.COLORMAP_VIRIDIS)[:, 0]
cv2.circle(img, (int(half / res),) * 2, 8, (0, 0, 255), -1)
cv2.imwrite(str(out / "floor.png"), img)

frame = {"camera_height": estimated * s, "estimated_camera_height": estimated, "scale": s,
         "floor_tilt_deg": float(np.degrees(np.arccos(n[2]))), "moge_merge_scale": scale,
         "R_moge_to_level": R.tolist(), "model": args.model, "width": w, "height": h}
(out / "frame.json").write_text(json.dumps(frame, indent=1))
print(f"{args.scene.name}: camera {estimated:.2f} m above the floor (estimated), floor tilt "
      f"{frame['floor_tilt_deg']:.1f} deg, max distance {distance[valid].max():.1f} m -> {out}")
