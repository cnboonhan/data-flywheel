"""Panorama geometry shared by the pano_arena steps (imported by each step's uv script; MoGe only where it's used).

Frames: MoGe's panorama frame has z up and the middle column looking along -x. The steps turn it 180 deg about z so the
middle column looks along +x ("panorama frame" below). The scene frame (step 1's frame.json) is the panorama frame
levelled with R_moge_to_level, with the origin on the floor below the camera.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def load_frame(scene: Path) -> dict:
    return json.loads((scene / "depth" / "frame.json").read_text())


def directions(h: int, w: int) -> np.ndarray:
    """Unit direction per pixel of an h x w equirectangular panorama, panorama frame."""
    v, u = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    theta, phi = (1 - u) * 2 * np.pi, v * np.pi
    return np.stack([-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)], -1)


def to_pixels(d: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Panorama pixel coordinates (x, y) of panorama-frame directions; the inverse of directions()."""
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    u = 1 - (np.arctan2(-d[..., 1], -d[..., 0]) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(d[..., 2], -1, 1)) / np.pi
    return (u * w - 0.5).astype(np.float32), (v * h - 0.5).astype(np.float32)


def points(distance: np.ndarray, frame: dict) -> np.ndarray:
    """Scene-frame point per panorama pixel."""
    R = np.array(frame["R_moge_to_level"])
    return (distance[..., None] * directions(*distance.shape)) @ R.T + [0, 0, frame["camera_height"]]


def camera(frame: dict, yaw: float, pitch: float) -> np.ndarray:
    """Rotation from a pinhole view (x right, y down, z forward) to the panorama frame; yaw/pitch in degrees, scene
    frame (yaw 0 = +x, positive to the left; pitch positive up)."""
    a, b = np.radians(yaw), np.radians(pitch)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
    view_to_scene = Rz @ Ry @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])   # columns: image right, down, forward
    return np.array(frame["R_moge_to_level"]).T @ view_to_scene


def view(frame: dict, h: int, w: int, yaw: float, pitch: float, size: int, fov: float = 90.0):
    """Remap grids (panorama pixel coordinates per view pixel) of a square pinhole view."""
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    x, y = np.meshgrid(np.arange(size) + 0.5 - size / 2, np.arange(size) + 0.5 - size / 2)
    return to_pixels(np.stack([x, y, np.full_like(x, f)], -1) @ camera(frame, yaw, pitch).T, h, w)


def project(frame: dict, d: np.ndarray, yaw: float, pitch: float, size: int, fov: float):
    """View pixel coordinates (x, y) of panorama-frame directions d, and whether each lies in front of the view."""
    c = d @ camera(frame, yaw, pitch)
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    z = np.where(c[..., 2] > 1e-6, c[..., 2], 1e-6)
    return ((c[..., 0] / z * f + size / 2 - 0.5).astype(np.float32),
            (c[..., 1] / z * f + size / 2 - 0.5).astype(np.float32), c[..., 2] > 1e-6)


def moge_distance(image: np.ndarray, model_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Metric distance per panorama pixel, and where it's valid. MoGe-2 runs on MoGe's 20-view panorama split; MoGe's
    gradient-domain merge loses the absolute scale, so the merged map is rescaled to the views' metric distances."""
    import torch
    import utils3d_moge as utils3d
    from moge.model.v2 import MoGeModel
    from moge.utils.panorama import (get_panorama_cameras, merge_panorama_depth, spherical_uv_to_directions,
                                     split_panorama_image)

    h, w = image.shape[:2]
    extr, intr = get_panorama_cameras()
    views = split_panorama_image(image, extr, intr, 512)
    model = MoGeModel.from_pretrained(model_name).cuda().eval()
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
    merged *= float(np.median(np.concatenate(ratios)))
    distance = cv2.resize(merged, (w, h), interpolation=cv2.INTER_LINEAR)
    return distance, cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
