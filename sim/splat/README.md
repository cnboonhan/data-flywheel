# sim/splat

Gaussian splats of real environments: COLMAP images in, a `ParticleField` USD out that Isaac Sim 6.x (and IsaacLab-Arena) renders as a background. Training is [3DGRUT](https://github.com/nv-tlabs/3dgrut) (3DGUT + MCMC, the trainer behind NVIDIA NuRec), an untouched submodule at `sim/splat/3dgrut`. `train.py` runs it on a local directory or `s3://` prefix and writes the run to either. Data lives in `datasets/splat/` (gitignored): local scenes, runs, the S3 cache.

## Install (once)

Install 3DGRUT into its venv (~20 GB in `sim/splat/3dgrut/.venv`, x86_64 or aarch64). On the cluster, run it on a GPU node with many cores.

```bash
bash sim/splat/install.sh
```

Notes:
- The CUDA toolkit lands in `.venv/cuda-<ver>/`, so the host needs no `nvcc`. On aarch64 (no upstream runfile) it is assembled from NVIDIA's redist tarballs.
- CUDA 12.8 by default, 13.0 for sm_103 (GB300), detected from the GPU. On a machine without the target GPU, set `CUDA_VERSION=12.8|13`.
- PyPI has no aarch64 `usd-core`, so there OpenUSD (core + Python) is built into `.venv/opt/usd` (~10 min).
- `DISPLAY` is unset during the install: the CUDA runfile otherwise tries to open an xterm (`exec: -title: not found`).
- The venv has absolute paths baked in; don't move the checkout afterwards.

## Run: zh_lounge

NVIDIA's Zurich office lounge from [`nvidia/PhysicalAI-Robotics-NuRec`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec) (gated: accept the terms, then `hf auth login`): 374 photos with COLMAP poses, 0.65 GB.

1. Fetch it.
   ```bash
   uvx hf download nvidia/PhysicalAI-Robotics-NuRec --repo-type dataset --include 'zh_lounge/*' --local-dir datasets/splat/zh_lounge
   ```
2. Train and export (every 8th image is held out and scored; add `n_iterations=7000` for a ~5 min check).
   ```bash
   uv run sim/splat/train.py datasets/splat/zh_lounge/zh_lounge/colmap datasets/splat/zh_lounge/runs
   ```

`train.py <input> <output> [3DGRUT overrides]` runs upstream's `train.py` with `apps/colmap_3dgut_mcmc.yaml` (`--config` to change it) and USD export on, and unzips `images.zip` when there is no `images/`.

### With the S3 gateway

Set up the S3 client ([versitygw](../../services/versitygw/README.md#client-setup)), then use `s3://` prefixes on either side.

```bash
uv run sim/splat/train.py s3://raw/open_datasets/nurec-zh_lounge/zh_lounge/colmap s3://processed/splats/zh_lounge
uv run sim/splat/train.py s3://raw/internal_datasets/real2sim/ridgeback_demo/colmap s3://processed/splats/ridgeback_demo --floor
```

An S3 input is mirrored into `datasets/splat/cache/inputs/` (only changed files are fetched); an S3 output is trained in `datasets/splat/cache/runs/` and uploaded to `<output>/<name>/<run>/`. Captures from [`sensors/real2sim`](../../sensors/real2sim/README.md) go to `s3://raw/internal_datasets/real2sim/<name>/colmap/`; `download-datasets-hf` fetches zh_lounge into `s3://raw/open_datasets/nurec-zh_lounge/`.

## On the Slurm cluster

Source `slurm.env` ([slurm/](../../services/slurm/README.md)), then submit `train.sbatch` with `train.py`'s arguments. It runs on one GPU and skips C03, whose GPUs Triton holds.

```bash
cd $STATE_DIR/slurm-logs
sbatch --export=ALL $FLYWHEEL_ROOT/sim/splat/train.sbatch \
  s3://raw/internal_datasets/real2sim/<scene>/colmap s3://processed/splats/<scene> --name <scene> --floor
```

## Output

`<output>/<name>/<run>/`: `export_last_lightfield.usdz` (one `ParticleField3DGaussianSplat` prim), `splat.usdc`, `scene.usda`, `metrics.json` (held-out PSNR/SSIM/LPIPS), checkpoints.

Open or reference **`scene.usda`**, not the USDZ:
- It places the splat in the COLMAP world frame, Z-up. The USDZ has the exporter's normalizing transform (cameras centred, Y-up) and cameras with a wrong field of view.
- It references `splat.usdc`, the export without gaussians beyond `--crop_radius` (30 m) of the median. 3DGRUT's background shell reaches ~1000 km and in Isaac Sim blocks the dome light (meshes render black on top and flicker).
- `--floor` adds an invisible collision floor at z = 0 under the cameras' footprint plus 2 m (needs a text COLMAP model). The splat has no other collision; add proxies or a mesh.
- The frame is metric and gravity-aligned only if the poses are, as in real2sim captures (the `map` frame). Plain SfM datasets like zh_lounge need placing by hand.

As an Arena background: [Splat backgrounds](../isaaclab_arena/README.md#splat-backgrounds).

## Notes

- **Large captures:** with 500+ images `train.py` lowers MCMC's opacity and scale penalties to 0.001 (override with `loss.lambda_opacity=` / `loss.lambda_scale=`). At 0.01 a 1008-image capture collapsed (11 dB held out); at 0.001 it trained (18 dB).
- **Gaussian cap:** `strategy.add.max_n_gaussians` (1M) limits growth only; a denser starting point cloud keeps its size (ridgeback_demo started and stayed at ~2M). 3M ran out of memory on a 24 GB GPU.
- **Settings sweep** (zh_lounge, 30k iterations, GB300, 2026-10-10; held-out PSNR / SSIM / LPIPS): the defaults, 27.27 / 0.921 / 0.339, were not beaten. 60k iterations 27.25; penalties 0.001 27.17; PPISP 27.19; 2-3M gaussians 26.8-27.2 (slightly sharper, worse held out). On ridgeback_demo, 3M gave 13.7 dB against 18.1 at the default, with two thirds of the gaussians beyond the 30 m crop. Quality is limited by the capture, not training.
- **Results:** zh_lounge at 7k iterations, 24.9 dB / 0.89 SSIM in ~5 min (RTX 5090 Laptop) and 24.8 dB in 8 min on a GB300 node through the S3 gateway, including the first-run JIT compile; at 30k, 27.2 dB / 0.92 in 31 min (laptop), 19 min (GB300).

## Your own scenes

Capture any COLMAP dataset (`images/` + `sparse/0/`):
- lock focus, exposure and white balance;
- 300–500 overlapping stills per room, at the heights the robot's cameras will see;
- no people or moving objects; clear what the robot will manipulate (added as separate assets);
- a few markers of known size for metric scale.

Poses come from COLMAP or GLOMAP (not set up here yet), or from [`sensors/real2sim`](../../sensors/real2sim/README.md).

## Planned: automated workcell capture

Not built yet. A robot maps the workcell with SLAM + navigation, then iterates: capture from map-planned viewpoints → train → find poorly reconstructed regions (low held-out PSNR, floaters, unobserved space) → plan new viewpoints. The map supplies metric scale and gravity alignment, and its poses can seed or replace COLMAP.
