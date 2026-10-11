# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "opencv-python-headless", "imageio-ffmpeg"]
# ///
"""A GIF of a trained splat: the camera circles its viewpoint and turns once around, rendered by 3DGRUT on this
machine's GPU (as train.py, after the Install in README.md).

    uv run sim/colmap_splat/flythrough.py <run dir> <out.gif> [--centre x y z] [--radius 0.2] [--frames 36]

<run dir> is a train.py run (ckpt_last.pt, parsed.yaml). The circle's centre is the mean of its dataset's camera
centres (a pano_ideate splat's capture point) unless --centre is given, in the run's frame (z up).
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from from_pano import quaternion, rotation  # noqa: E402

GRUT = HERE / "3dgrut"
SPLIT = 8          # 3DGRUT renders every 8th view of a dataset (its test_split_interval)
SIZE, FOV = 640, 80.0   # px and degrees of the rendered views: a wide room view at GIF size (untuned)


def centres(images_txt: Path) -> np.ndarray:
    """Camera centres of a text COLMAP model."""
    out = []
    for line in images_txt.read_text().splitlines():
        f = line.split()
        if len(f) >= 10 and f[-1].lower().endswith((".jpg", ".png", ".jpeg")):
            qw, qx, qy, qz, tx, ty, tz = map(float, f[1:8])
            R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                          [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                          [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            out.append(-R.T @ [tx, ty, tz])
    return np.array(out)


def path_dataset(root: Path, centre: np.ndarray, radius: float, frames: int, pitch: float = -8.0):
    """A stand-in COLMAP dataset whose every SPLIT-th view is a frame of the turn (blank images: render.py scores
    against them, the scores mean nothing)."""
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    cv2.imwrite(str(root / "blank.jpg"), np.zeros((SIZE, SIZE, 3), np.uint8))
    f = 0.5 * SIZE / math.tan(math.radians(FOV) / 2)
    (root / "sparse/0/cameras.txt").write_text(f"1 PINHOLE {SIZE} {SIZE} {f:.4f} {f:.4f} {SIZE / 2} {SIZE / 2}\n")
    (root / "sparse/0/points3D.txt").write_text(f"1 {centre[0]:.3f} {centre[1]:.3f} {centre[2]:.3f} 0 0 0 0\n")
    lines, n = [], frames * SPLIT
    for k in range(n):
        t = k / n
        c = centre + radius * np.array([math.cos(2 * math.pi * t), math.sin(2 * math.pi * t), 0.0])
        Rcw = rotation(360 * t, pitch).T
        q, tv = quaternion(Rcw), -Rcw @ c
        (root / "images" / f"p_{k:05d}.jpg").symlink_to(root / "blank.jpg")
        lines += [f"{k + 1} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.6f}' for v in tv)} 1 p_{k:05d}.jpg", ""]
    (root / "sparse/0/images.txt").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path, help="a train.py run directory")
    p.add_argument("out", type=Path, help="GIF to write")
    p.add_argument("--centre", type=float, nargs=3, help="x y z of the circle's centre (default: mean camera centre)")
    p.add_argument("--radius", type=float, default=0.2, help="m; small, since a pano splat holds up within ~0.6 m")
    p.add_argument("--frames", type=int, default=36, help="frames per turn (36: a 10 deg step)")
    a = p.parse_args()
    run = a.run.resolve()
    data = Path(re.search(r"^path:\s*(\S+)", (run / "parsed.yaml").read_text(), re.M).group(1))
    if a.centre:
        centre = np.array(a.centre)
    elif (data / "sparse/0/images.txt").exists():
        centre = centres(data / "sparse/0/images.txt").mean(0)
    else:
        sys.exit("the dataset has no text COLMAP model to take the centre from: give --centre x y z")
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        tmp = Path(tmp)
        path_dataset(tmp / "path", centre, a.radius, a.frames)
        env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
        subprocess.run(["bash", "-c", 'source .venv/bin/activate && exec python render.py "$@"', "render",
                        "--checkpoint", str(run / "ckpt_last.pt"), "--path", str(tmp / "path"),
                        "--out-dir", str(tmp / "out")], cwd=GRUT, env=env, check=True, stdout=subprocess.DEVNULL)
        renders = next((tmp / "out").rglob("renders"))
        import imageio_ffmpeg
        a.out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-framerate", "5", "-i",
                        str(renders / "%05d.png"), "-vf", "scale=400:-1:flags=lanczos,split[x][y];"
                        "[x]palettegen=max_colors=96[p];[y][p]paletteuse=dither=bayer:bayer_scale=5", str(a.out)],
                       check=True)
    print(f"{a.frames} frames around {np.round(centre, 2).tolist()} -> {a.out}")


if __name__ == "__main__":
    main()
