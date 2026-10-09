# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3>=1.34"]
# ///
"""Train a Gaussian splat with 3DGRUT; input and output can each be a local directory or an s3:// prefix.

    uv run sim/splat/train.py <input> <output> [3DGRUT hydra overrides...] [--floor]
    uv run sim/splat/train.py datasets/splat/zh_lounge/zh_lounge/colmap datasets/splat/zh_lounge/runs n_iterations=7000
    uv run sim/splat/train.py s3://raw/open_datasets/nurec-zh_lounge/zh_lounge/colmap s3://processed/splats/zh_lounge

<input> is a COLMAP dataset: sparse/0/*.bin plus images/ (or images.zip, unzipped on first use). An s3:// input is
synced into --cache (only missing or changed files are fetched). The run is written under <output>/<name>/<run>/;
for an s3:// output it is trained in --cache and then uploaded. S3 settings are the services stack's: S3_ENDPOINT_URL
plus AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION and AWS_CA_BUNDLE (see docs/flywheel.md).
Each run also gets scene.usda: the splat in the COLMAP world frame, Z-up (--floor adds a collision floor at z = 0).
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRUT = HERE / "3dgrut"


def split_s3(url: str) -> tuple[str, str]:
    bucket, _, prefix = url.removeprefix("s3://").partition("/")
    return bucket, prefix.strip("/")


def s3_client():
    import boto3

    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"))


def download(url: str, dst: Path) -> None:
    """Mirror s3://bucket/prefix into dst, skipping files whose size already matches."""
    s3, (bucket, prefix) = s3_client(), split_s3(url)
    n = fetched = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) + 1:]
            if not rel or rel.endswith("/"):
                continue
            n += 1
            local = dst / rel
            if local.exists() and local.stat().st_size == obj["Size"]:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, obj["Key"], str(local))
            fetched += 1
    if n == 0:
        sys.exit(f"nothing under {url}")
    print(f"input: {url} -> {dst} ({fetched} of {n} files fetched)")


def upload(src: Path, url: str) -> None:
    s3, (bucket, prefix) = s3_client(), split_s3(url)
    files = [p for p in src.rglob("*") if p.is_file()]
    for p in files:
        s3.upload_file(str(p), bucket, f"{prefix}/{p.relative_to(src).as_posix()}".lstrip("/"))
    print(f"output: {src} -> {url} ({len(files)} files)")


SCENE_USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{
    def "splat" (
        prepend references = @./%s@</World/gaussians/Gaussians/gaussians>
    )
    {
    }
%s}
"""

# Invisible static collider whose top face is z = 0; only meaningful when the COLMAP world is gravity-aligned and
# metric with the floor at z = 0, as in sensors/real2sim captures (map frame).
FLOOR = """
    def Cube "floor" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        token visibility = "invisible"
        double3 xformOp:translate = (0, 0, -0.05)
        float3 xformOp:scale = (500, 500, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
"""


def write_scene(run: Path, floor: bool) -> None:
    """scene.usda: the splat in the COLMAP world frame, Z-up. The export puts a normalizing transform (cameras
    centred, Y-up) on the splat's parent Xform; referencing the splat prim itself drops it."""
    usdz = sorted(run.glob("export_last*.usdz"))
    if usdz:
        (run / "scene.usda").write_text(SCENE_USDA % (usdz[-1].name, FLOOR if floor else ""))
        print(f"scene: {run / 'scene.usda'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="COLMAP dataset dir or s3://bucket/prefix")
    parser.add_argument("output", help="dir or s3://bucket/prefix that receives <name>/<run>/")
    parser.add_argument("overrides", nargs="*", help="3DGRUT hydra overrides, e.g. n_iterations=7000")
    parser.add_argument("--name", help="experiment name (default: the input's parent dir name)")
    parser.add_argument("--floor", action="store_true",
                        help="add a collision floor at z = 0 to scene.usda (gravity-aligned metric input, e.g. real2sim)")
    parser.add_argument("--config", default="apps/colmap_3dgut_mcmc.yaml", help="3DGRUT config")
    parser.add_argument("--cache", type=Path, default=ROOT / "datasets" / "splat" / "cache",
                        help="local mirror of s3:// inputs and outputs")
    args = parser.parse_args()

    python = GRUT / ".venv" / "bin" / "python"
    if not python.exists():
        sys.exit("3DGRUT not installed; see sim/splat/README.md (Install)")

    if args.input.startswith("s3://"):
        data = args.cache / "inputs" / "/".join(split_s3(args.input))
        download(args.input, data)
    else:
        data = Path(args.input).resolve()
    if not (data / "sparse" / "0").is_dir():
        sys.exit(f"{data} has no sparse/0 (COLMAP model)")
    if not (data / "images").is_dir():
        if not (data / "images.zip").exists():
            sys.exit(f"{data} has neither images/ nor images.zip")
        with zipfile.ZipFile(data / "images.zip") as z:   # archives contain images/...
            z.extractall(data)
    name = args.name or (data.parent.name if data.name == "colmap" else data.name)

    to_s3 = args.output.startswith("s3://")
    out = args.cache / "runs" / "/".join(split_s3(args.output)) if to_s3 else Path(args.output).resolve()
    before = set((out / name).glob("*")) if (out / name).exists() else set()

    # The venv's activate script exports its bundled CUDA toolkit, which the tracer's JIT compile needs.
    cmd = ["bash", "-c", 'source .venv/bin/activate && exec python train.py "$@"', "train",
           f"--config-name={args.config}", f"path={data}", f"out_dir={out}", f"experiment_name={name}",
           "export_usd.enabled=true", *args.overrides]
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    subprocess.run(cmd, cwd=GRUT, env=env, check=True)

    new_runs = sorted(set((out / name).glob("*")) - before, key=lambda p: p.stat().st_mtime)
    if not new_runs:
        sys.exit(f"training finished but no new run under {out / name}")
    run = new_runs[-1]
    print(f"run: {run}")
    write_scene(run, args.floor)
    if to_s3:
        upload(run, f"{args.output.rstrip('/')}/{name}/{run.name}")


if __name__ == "__main__":
    main()
