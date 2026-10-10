# /// script
# requires-python = ">=3.10"
# dependencies = ["moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebe"]
# ///
"""Step 1: metric depth of an equirectangular panorama, and the frame the rest of pano_arena works in.

    uv run sim/pano_arena/depth.py datasets/pano_arena/<scene> [--camera_height 1.5]

Reads <scene>/source.jpg. MoGe-2 metric depth (common.moge_distance). The floor is the lowest large horizontal layer of points. The output frame is Arena's: z up, origin on the floor
below the camera, +x towards the middle of the panorama. --camera_height rescales to a known height above the floor.
Writes <scene>/depth/: distance.npy (metres from the camera, per panorama pixel), pano.jpg (the image at that size),
points.ply (coloured points in the output frame), depth.png and floor.png (previews), frame.json.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from moge.utils.vis import colorize_depth

from common import directions, moge_distance

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

distance, valid = moge_distance(image, args.model)
pts = distance[..., None] * directions(h, w)

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
         "floor_tilt_deg": float(np.degrees(np.arccos(n[2]))),
         "R_moge_to_level": R.tolist(), "model": args.model, "width": w, "height": h}
(out / "frame.json").write_text(json.dumps(frame, indent=1))
print(f"{args.scene.name}: camera {estimated:.2f} m above the floor (estimated), floor tilt "
      f"{frame['floor_tilt_deg']:.1f} deg, max distance {distance[valid].max():.1f} m -> {out}")
