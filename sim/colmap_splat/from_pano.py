# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "tritonclient[http]"]
# ///
"""A COLMAP dataset of the background of a pano_ideate session (its room with the interacted objects erased), to
train with train.py like any capture.

    uv run sim/colmap_splat/from_pano.py <scene>_session.zip datasets/colmap_splat/<scene>
    uv run sim/colmap_splat/train.py datasets/colmap_splat/<scene> datasets/colmap_splat/<scene>/runs --floor

The model is in the session's scene frame (metric, z up, origin on the floor below the capture point), so --floor and
Arena placement work as for a real2sim capture. Views: 26 from the capture point, taken from the panorama, and 12 from
each of 24 points up to 0.6 m around it, drawn from the panorama's depth; what the panorama never saw is filled in,
by OpenCV, or with --fill editor by Qwen-Image-Edit at EDIT_ENDPOINT (services/triton/triton_endpoint.py; ~10 s a
view). Points: 600k from the depth. Writes <out>/images/ and <out>/sparse/0/ (text).
"""

import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

LOOKS = [(y, p) for p in (-60, -20, 20) for y in range(0, 360, 45)] + [(0, -90), (0, 90)]    # yaw, pitch (deg)
AROUND = [(r * np.cos(np.radians(a)), r * np.sin(np.radians(a)), dz)
          for r in (0.3, 0.6) for a in range(0, 360, 60) for dz in (-0.2, 0.15)]                # offsets (m)


def rotation(yaw: float, pitch: float) -> np.ndarray:
    """View (x right, y down, z forward) to scene; yaw from +x towards +y, pitch up."""
    a, b = np.radians(yaw), np.radians(pitch)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
    return Rz @ Ry @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])


def points(distance: np.ndarray, frame: dict) -> np.ndarray:
    """Scene-frame point per panorama pixel (the panorama's middle column looks along +x)."""
    h, w = distance.shape
    v, u = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    theta, phi = (1 - u) * 2 * np.pi, v * np.pi
    d = np.stack([-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)], -1)
    return (d * distance[..., None]) @ np.array(frame["R_moge_to_level"]).T + [0, 0, frame["camera_height"]]


def render(xyz: np.ndarray, rgb: np.ndarray, centre: np.ndarray, R: np.ndarray, size: int, f: float):
    """The points seen from centre (nearest wins), and where nothing landed."""
    c = (xyz - centre) @ R
    ok = c[:, 2] > 0.05
    x, y = c[ok, 0] / c[ok, 2] * f + size / 2, c[ok, 1] / c[ok, 2] * f + size / 2
    inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    xi, yi, z, col = x[inside].astype(int), y[inside].astype(int), c[ok, 2][inside], rgb[ok][inside]
    order = np.argsort(-z)
    img, depth = np.zeros((size, size, 3), np.uint8), np.full((size, size), np.inf, np.float32)
    img[yi[order], xi[order]], depth[yi[order], xi[order]] = col[order], z[order]
    return img, ~np.isfinite(depth)


def fill_editor(edit, img: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Qwen-Image-Edit (edit: its triton_endpoint.Endpoint) inpaints the holes (shown in magenta), pasted back only
    there."""
    shown = np.where(holes[..., None], [255, 0, 255], img).astype(np.uint8)
    out = edit.infer({"images": [cv2.imencode(".png", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR))[1].tobytes()],
                      "prompt": [b"Fill the magenta areas with what continues the room around them: walls, floor, "
                                 b"furniture surfaces, lighting. Add no objects. Keep everything else as it is."],
                      "steps": np.array([20], np.int32)}, ["image"])["image"]
    res = cv2.cvtColor(cv2.imdecode(np.frombuffer(np.ravel(out)[0], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    res = cv2.resize(res, img.shape[1::-1], interpolation=cv2.INTER_AREA)
    return np.where(cv2.dilate(holes.astype(np.uint8), np.ones((5, 5), np.uint8))[..., None] > 0, res, img)


def run(session: Path, out: Path, size: int = 1024, fov: float = 90.0, n_points: int = 600000, fill: str = "opencv",
        log=print):
    with zipfile.ZipFile(session) as z:
        frame = json.loads(z.read("scene.json"))["frame"]
        pano = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read("background/pano.jpg"), np.uint8), cv2.IMREAD_COLOR),
                            cv2.COLOR_BGR2RGB)
        distance = np.load(io.BytesIO(z.read("background/distance.npy")))
    pano = cv2.resize(pano, distance.shape[::-1], interpolation=cv2.INTER_AREA)
    shutil.rmtree(out / "images", ignore_errors=True)
    shutil.rmtree(out / "sparse", ignore_errors=True)
    (out / "images").mkdir(parents=True)
    (out / "sparse" / "0").mkdir(parents=True)
    eye = np.array([0, 0, frame["camera_height"]])
    xyz, rgb = points(distance, frame), pano
    f = 0.5 * size / np.tan(np.radians(fov) / 2)
    dense = xyz[::2, ::2].reshape(-1, 3), rgb[::2, ::2].reshape(-1, 3)       # enough points for 1024-px views
    full = xyz.reshape(-1, 3), rgb.reshape(-1, 3)
    cams = [(f"c_{k:02d}.jpg", eye, rotation(*look), full) for k, look in enumerate(LOOKS)]   # exact: every pixel
    cams += [(f"o_{n:02d}_{k:02d}.jpg", eye + off, rotation(*look), dense)
             for n, off in enumerate(AROUND) for k, look in enumerate(LOOKS[:-2:2])]
    edit = None
    if fill == "editor":
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "triton"))
        from triton_endpoint import Endpoint
        edit = Endpoint("EDIT", "Qwen-Image-Edit-2511")
        log(f"{edit}")
    lines = []
    for i, (name, centre, R, (pts, col)) in enumerate(cams, 1):
        img, holes = render(pts, col, centre, R, size, f)
        if name.startswith("c_"):                                   # from the capture point nothing is hidden
            img = cv2.inpaint(img, holes.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        elif edit and holes.mean() > 0.01:
            img = fill_editor(edit, cv2.inpaint(img, holes.astype(np.uint8), 3, cv2.INPAINT_TELEA), holes)
        else:
            img = cv2.inpaint(img, holes.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        cv2.imwrite(str(out / "images" / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        Rcw = R.T                                                   # scene -> camera
        t = -Rcw @ centre
        q = quaternion(Rcw)
        lines += [f"{i} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.6f}' for v in t)} 1 {name}", ""]
        if i % 50 == 0:
            log(f"  {i}/{len(cams)} views")
    sparse = out / "sparse" / "0"
    (sparse / "cameras.txt").write_text(f"1 PINHOLE {size} {size} {f:.6f} {f:.6f} {size / 2} {size / 2}\n")
    (sparse / "images.txt").write_text("\n".join(lines) + "\n")
    keep = np.random.default_rng(0).choice(xyz.shape[0] * xyz.shape[1], n_points, replace=False)
    p, c = xyz.reshape(-1, 3)[keep], rgb.reshape(-1, 3)[keep]
    with open(sparse / "points3D.txt", "w") as fh:
        for i, (a, b) in enumerate(zip(p, c), 1):
            fh.write(f"{i} {a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {b[0]} {b[1]} {b[2]} 0\n")
    log(f"{len(cams)} views ({len(LOOKS)} from the capture point), {n_points} points -> {out}")


def quaternion(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("session", type=Path, help="a session zip from pano_ideate's viewer (Download session)")
    p.add_argument("out", type=Path, help="dataset folder to write (images/, sparse/0/)")
    p.add_argument("--fill", choices=["opencv", "editor"], default="opencv",
                   help="what fills what the panorama never saw in the views around it")
    a = p.parse_args()
    run(a.session, a.out, fill=a.fill)
