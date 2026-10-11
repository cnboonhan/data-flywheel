# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "trimesh", "tritonclient[http]", "transformers", "jinja2"]
# ///
"""Step 1: metric depth and the scene frame of an equirectangular panorama.

    uv run sim/pano_ideate/depth.py datasets/pano_ideate/<scene>

Reads <scene>/source.jpg. MoGe-3 (ViT-g, MIT) on Triton gives the metric distance per pixel (20 perspective views,
merged at the panorama's full width: the slow part, a few minutes, runs on the server). The floor is the lowest large
horizontal layer of points. The output frame is Arena's: z up, origin on the floor below the camera, +x towards the
middle of the panorama.
Writes <scene>/depth/: pano.jpg (the panorama at --width), distance.npy (metres per pixel), points.ply (coloured points
in the output frame), depth.png and floor.png (previews), frame.json.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import models
from common import directions


def run(scene: Path, triton: models.Triton, camera_height: float | None = None, width: int = 4096,
        view_size: int = 1024, merge_width: int = 0, log=print) -> dict:
    out = scene / "depth"
    out.mkdir(exist_ok=True)
    image = cv2.cvtColor(cv2.imread(str(scene / "source.jpg")), cv2.COLOR_BGR2RGB)
    if abs(image.shape[1] - 2 * image.shape[0]) > 2:
        raise ValueError(f"{image.shape[1]}x{image.shape[0]} is not an equirectangular panorama (2:1)")
    image = cv2.resize(image, (width, width // 2), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]

    distance, valid = triton.depth(image, 0, view_size, merge_width)
    pts = distance[..., None] * directions(h, w)

    # Floor: the lowest z layer (2 cm bins) holding at least 2% of the points below the camera, refined with a plane.
    below = pts[valid & (pts[..., 2] < -0.3)]
    hist, edges = np.histogram(below[:, 2], bins=np.arange(below[:, 2].min(), -0.3, 0.02))
    layer = np.flatnonzero(hist > 0.02 * len(below))[0]
    z0 = edges[layer:layer + 2].mean()
    layer_pts = below[np.abs(below[:, 2] - z0) < 0.05]
    c = layer_pts.mean(0)
    n = np.linalg.svd(layer_pts - c, full_matrices=False)[2][2]
    n = n if n[2] > 0 else -n
    estimated = float(-n @ c)                                # camera's height above the floor plane

    # Level the floor (rotate n onto +z), put the origin under the camera, rescale to camera_height if given.
    axis = np.cross(n, [0, 0, 1])
    R = cv2.Rodrigues(axis / (np.linalg.norm(axis) + 1e-12) * np.arcsin(min(np.linalg.norm(axis), 1.0)))[0]
    s = camera_height / estimated if camera_height else 1.0
    pts = (pts @ R.T) * s
    pts[..., 2] += estimated * s
    distance = distance * s

    np.save(out / "distance.npy", distance.astype(np.float32))
    cv2.imwrite(str(out / "pano.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    inv = np.where(valid, 1 / np.maximum(distance, 1e-3), 0)
    lo, hi = np.percentile(inv[valid], [1, 99])
    cv2.imwrite(str(out / "depth.png"), cv2.applyColorMap((np.clip((inv - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                                                          cv2.COLORMAP_TURBO))
    step = max(1, w // 1024)
    sel = valid[::step, ::step]
    import trimesh
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
             "R_moge_to_level": R.tolist(), "model": triton.depth_.model, "width": w, "height": h}
    (out / "frame.json").write_text(json.dumps(frame, indent=1))
    log(f"{scene.name}: camera {estimated:.2f} m above the floor (estimated), floor tilt "
        f"{frame['floor_tilt_deg']:.1f} deg, max distance {distance[valid].max():.1f} m -> {out}")
    return frame


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("--camera_height", type=float, help="metres above the floor; default: as estimated")
    p.add_argument("--width", type=int, default=4096, help="panorama width to work at")
    p.add_argument("--view_size", type=int, default=1024, help="px, each of MoGe's 20 perspective views")
    p.add_argument("--merge_width", type=int, default=0, help="px, width the views' depths are merged at (0: --width)")
    a = p.parse_args()
    run(a.scene, models.Triton(), a.camera_height, a.width, a.view_size, a.merge_width)
