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

### Robot-posed captures: refine the poses first

Refine poses that come from a robot's localisation (as in [`sensors/real2sim`](../../sensors/real2sim/README.md)) against the images before training; a few cm and about a degree of pose error blur a splat more than any training setting.

```bash
uv run sim/splat/refine_poses.py datasets/real2sim/run1/colmap datasets/real2sim/run1/colmap_refined   # ~5 min for 1000 images
uv run sim/splat/train.py datasets/real2sim/run1/colmap_refined datasets/real2sim/run1/runs --floor
```

`refine_poses.py` matches each image with its nearest views (GPU SIFT), triangulates from the input poses and bundle-adjusts poses and points with the lenses fixed, then maps the result back onto the input camera centres, so it stays in the map frame at metric scale. Check its last lines: reprojection error should be well under 1 px median, and it prints how far the poses moved. It runs on x86_64 only (no aarch64 `pycolmap-cuda12` wheels), so run it on the capture machine and upload the refined model.

### With the S3 gateway

Set up the S3 client ([versitygw](../../services/versitygw/README.md#client-setup)), then use `s3://` prefixes on either side.

```bash
uv run sim/splat/train.py s3://raw/open_datasets/nurec-zh_lounge/zh_lounge/colmap s3://processed/splats/zh_lounge
uv run sim/splat/train.py s3://raw/internal_datasets/real2sim/ridgeback_demo/colmap_refined s3://processed/splats/ridgeback_demo --floor
```

An S3 input is mirrored into `datasets/splat/cache/inputs/` (only changed files are fetched); an S3 output is trained in `datasets/splat/cache/runs/` and uploaded to `<output>/<name>/<run>/`. Captures from [`sensors/real2sim`](../../sensors/real2sim/README.md) go to `s3://raw/internal_datasets/real2sim/<name>/colmap/`, refined ones next to them in `colmap_refined/` (sync its `sparse/` and `images/`, not `work/`); `download-datasets-hf` fetches zh_lounge into `s3://raw/open_datasets/nurec-zh_lounge/`.

## On the Slurm cluster

Source `slurm.env` ([slurm/](../../services/slurm/README.md)), then submit `train.sbatch` with `train.py`'s arguments. It runs on one GPU; exclude the service node (`SERVICE_NODE` in `services/.env`), whose GPUs Triton holds.

```bash
cd $STATE_DIR/slurm-logs
sbatch --export=ALL --exclude=$SERVICE_NODE $FLYWHEEL_ROOT/sim/splat/train.sbatch \
  s3://raw/internal_datasets/real2sim/<scene>/colmap_refined s3://processed/splats/<scene> --name <scene> --floor
```

## Output

`<output>/<name>/<run>/`: `export_last_lightfield.usdz` (one `ParticleField3DGaussianSplat` prim), `splat.usdc`, `scene.usda`, `export_last.ply` (the standard 3DGS PLY, for other splat viewers and tools; uncropped), `metrics.json` (held-out PSNR/SSIM/LPIPS), checkpoints.

To write the PLY for a run trained before PLY export was on, run this on a GPU node:

```bash
cd sim/splat/3dgrut && source .venv/bin/activate && python ../export_ply.py <run dir>
```

Open or reference **`scene.usda`**, not the USDZ:
- It places the splat in the COLMAP world frame, Z-up. The USDZ has the exporter's normalizing transform (cameras centred, Y-up) and cameras with a wrong field of view.
- It references `splat.usdc`, the export without gaussians beyond `--crop_radius` (30 m) of the median. 3DGRUT's background shell reaches ~1000 km and in Isaac Sim blocks the dome light (meshes render black on top and flicker).
- `--floor` adds an invisible collision floor at z = 0 under the cameras' footprint plus 2 m (needs a text COLMAP model). The splat has no other collision; add proxies or a mesh.
- The frame is metric and gravity-aligned only if the poses are, as in real2sim captures (the `map` frame). Plain SfM datasets like zh_lounge need placing by hand.

As an Arena background: [Splat backgrounds](../isaaclab_arena/README.md#splat-backgrounds).

## Notes

- **What to expect:** zh_lounge at 30k iterations scores ~27 dB PSNR / 0.92 SSIM held out, in ~19 min on a GB300 (~31 min on an RTX 5090 Laptop). At 7k iterations expect ~25 dB in ~5 min, plus a few minutes for the tracer's JIT compile on the first run on a machine. The real2sim ridgeback_demo capture (1008 images) at 7k iterations: 17.4 dB with a pinhole lens and localisation poses, 18.9 dB with the right lens, 33.1 dB / 0.96 SSIM with refined poses.
- **Keep the default settings.** 60k iterations, weaker penalties, PPISP and 2-3M gaussians did not improve held-out quality on zh_lounge; more gaussians make renders slightly sharper but worse on new views, and on a large capture most of the extra gaussians go to the background shell beyond the crop. Improve the capture instead (coverage, lens model, poses).
- **Large captures:** with 500+ images `train.py` lowers MCMC's opacity and scale penalties to 0.001, because the default 0.01 lets a 1000-image capture collapse. Override with `loss.lambda_opacity=` / `loss.lambda_scale=`.
- **Gaussian cap:** `strategy.add.max_n_gaussians` (1M) only limits growth; a denser starting point cloud (e.g. real2sim's lidar points) keeps its size. On a 24 GB GPU, stay at or below ~2M.

## Your own scenes

Capture any COLMAP dataset (`images/` + `sparse/0/`):
- lock focus, exposure and white balance;
- 300–500 overlapping stills per room, at the heights the robot's cameras will see;
- no people or moving objects; clear what the robot will manipulate (added as separate assets);
- a few markers of known size for metric scale.

Poses come from COLMAP or GLOMAP (not set up here yet), or from [`sensors/real2sim`](../../sensors/real2sim/README.md).

## Planned: automated workcell capture

Not built yet. A robot maps the workcell with SLAM + navigation, then iterates: capture from map-planned viewpoints → train → find poorly reconstructed regions (low held-out PSNR, floaters, unobserved space) → plan new viewpoints. The map supplies metric scale and gravity alignment, and its poses can seed or replace COLMAP.
