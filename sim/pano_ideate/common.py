"""Panorama geometry of pano_ideate (numpy only).

Frames: MoGe's panorama frame has z up and the middle column looking along -x. The steps turn it 180 deg about z so the
middle column looks along +x ("panorama frame" below). The scene frame (step 1's frame.json) is the panorama frame
levelled with R_moge_to_level, with the origin on the floor below the camera.
"""

import json
from pathlib import Path

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
    """Remap grids (panorama pixel coordinates per view pixel) of a square pinhole view. fov 90 deg by default:
    objects.py's views, 45 deg apart, overlap by half (untuned)."""
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
