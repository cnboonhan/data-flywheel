# sim/splat

Gaussian splats of real environments for simulation: images with camera poses in, a splat as a `ParticleField` USD
out, which Isaac Sim 6.x (and IsaacLab-Arena) renders as a background. Training is [3DGRUT](https://github.com/nv-tlabs/3dgrut)
(3DGUT + MCMC, the trainer behind NVIDIA NuRec's mono workflow), an untouched upstream submodule at `sim/splat/3dgrut`
installed with its own uv scripts. The only local code is `train.py`, a uv script that runs 3DGRUT's `train.py` on a
local directory or an `s3://` prefix and writes the run to either.

Data stays out of git: `datasets/splat/` (gitignored) holds local scenes, runs and the S3 cache.

## Install (once)

```bash
git submodule update --init --recursive sim/splat/3dgrut
cd sim/splat/3dgrut
env -u DISPLAY FORCE_LOCAL_CUDA=1 CUDA_VERSION=12 ./scripts/create_venv_cuda.sh 3dgrut   # CUDA 12.8 into .venv, ~4 GB download
source .venv/bin/activate && env -u DISPLAY ./install_env_uv.sh                           # ~10 min, ~17 GB venv
```

CUDA 12.8 is needed for Blackwell (RTX 50xx); the host has no `nvcc`, so it goes into the venv. `DISPLAY` must be
unset: the CUDA runfile is a makeself archive that tries to open an xterm when it sees a display but no TTY
(`exec: -title: not found`). The venv has absolute paths baked in, so don't move the checkout after installing.

## Run: zh_lounge

NVIDIA's Zurich office lounge from [`nvidia/PhysicalAI-Robotics-NuRec`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec)
(gated: accept the terms, then `hf auth login`): 374 Nikon Z7 photos with COLMAP poses, plus NVIDIA's own reconstruction for comparison. 0.65 GB.

```bash
# from the repo root
# 1. Fetch
uvx hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset --include 'zh_lounge/*' --local-dir datasets/splat/zh_lounge

# 2. Prepare: 3DGRUT reads COLMAP as images/ next to sparse/0/
unzip -q -o datasets/splat/zh_lounge/zh_lounge/colmap/images.zip -d datasets/splat/zh_lounge/zh_lounge/colmap

# 3. Train + export (every 8th image held out and scored; the first run also JIT-compiles the tracer)
uv run sim/splat/train.py datasets/splat/zh_lounge/zh_lounge/colmap datasets/splat/zh_lounge/runs
#                                                                    add n_iterations=7000 for a ~5 min check
```

`train.py` takes `<input> <output> [3DGRUT overrides]` and calls upstream `train.py` with
`--config-name apps/colmap_3dgut_mcmc.yaml export_usd.enabled=true` (`--config` to change it). It unzips
`images.zip` when there is no `images/`, so step 2 is optional.

### With the S3 gateway

Either side can be an `s3://` prefix. The script uses the services stack's settings ([docs/flywheel.md](../../docs/flywheel.md)):

```bash
export S3_ENDPOINT_URL=https://s3.<host>:8443 AWS_ACCESS_KEY_ID=<name> AWS_SECRET_ACCESS_KEY=<secret> \
       AWS_DEFAULT_REGION=us-east-1 AWS_CA_BUNDLE=flywheel-ca.crt
uv run sim/splat/train.py s3://raw/open_datasets/nurec-zh_lounge/zh_lounge/colmap s3://processed/splats/zh_lounge
```

An S3 input is mirrored into `datasets/splat/cache/inputs/` (only missing or changed files are fetched, so reruns
start immediately); an S3 output is trained in `datasets/splat/cache/runs/` and the run directory is uploaded to
`<output>/<name>/<run>/`. The `download-datasets-hf` workflow already lists `zh_lounge` for `s3://raw/open_datasets/nurec-zh_lounge/`.

Verified (2026-10-09) against a local S3 server (moto) laid out like the gateway: input mirrored from
`s3://raw/open_datasets/nurec-zh_lounge/zh_lounge/colmap/` (4 of 4 files fetched, 0 of 4 on a rerun), run uploaded to
`s3://processed/splats/zh_lounge/zh_lounge/<run>/` (52 objects including the USDZ and `metrics.json`). Not yet run
against the real gateway. The local path was verified with 7k iterations: 25.0 dB PSNR / 0.89 SSIM.

Output: `datasets/splat/zh_lounge/runs/zh_lounge/<run>/` with `export_last_lightfield.usdz` (the splat, one
`ParticleField3DGaussianSplat` prim), `scene.usda`, `metrics.json` (held-out PSNR/SSIM/LPIPS) and checkpoints.

**Sim-ready scene:** open or reference `scene.usda`, not the USDZ. It places the splat in the COLMAP world frame, Z-up. The USDZ
itself has the exporter's normalizing transform (cameras centred, Y-up) and cameras with a wrong field of view. The COLMAP
frame is metric and gravity-aligned only if the poses are, as in [`sensors/real2sim`](../../sensors/real2sim/README.md)
captures (the `map` frame). Plain SfM datasets like zh_lounge have arbitrary scale and orientation, so place them by hand.
`--floor` adds an invisible collision floor at z = 0. The splat has no other collision; add proxies or a mesh. An Arena
background is a `LibraryBackground` subclass with `usd_path` pointing at `scene.usda`, loaded through Arena's
`--external_environment_class_path` hook.

**Reference results** (RTX 5090 Laptop, this commit): 7k iterations, 24.9 dB PSNR / 0.89 SSIM in ~5 min;
30k iterations (default), 27.2 dB / 0.92 SSIM in 31 min.

## Your own scenes

Any COLMAP dataset works the same way (`images/` + `sparse/0/`). Capture: lock focus, exposure and white balance;
300–500 overlapping stills per room at the heights the robot's cameras will see; no people or moving objects; clear
the things the robot will manipulate (they get added as separate assets); a few markers of known size for metric scale.
Poses come from COLMAP or GLOMAP (not set up here yet).

## Planned: automated workcell capture

Not built yet. A robot maps the workcell with a traditional SLAM + navigation stack, then visits viewpoints chosen
from that map to capture images iteratively: capture → train → find poorly reconstructed regions (low held-out PSNR,
floaters, unobserved space) → plan new viewpoints → capture again. The SLAM map supplies metric scale and gravity
alignment, and its poses can seed or replace COLMAP.
