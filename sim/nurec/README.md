# sim/nurec

How we build simulation scenes from real spaces with [NVIDIA Omniverse NuRec](https://docs.nvidia.com/nurec/): capture → Gaussian-splat reconstruction → sim-ready USD → IsaacLab-Arena environment. The same scene serves system 1 (the splat is the camera background, the mesh gives contact) and system 2 (the occupancy map gives navigation).

Verified on 2026-10-07 with Isaac Sim 6.1 (Arena's `.venv`) on the RTX 5090: NVIDIA's `nova_carter-wormhole` sample renders inside Arena with a Franka and a cube, the cube rests on the scene's collision mesh, and the splat occludes the robot correctly.

## What NuRec is

NuRec reconstructs camera (optionally lidar) captures into a 3D Gaussian splat and packages it as a USD scene that Isaac Sim renders in the RTX viewport, composed with ordinary meshes and physics. Two robotics workflows are documented:

| Workflow | Input | Output |
|---|---|---|
| [Mono camera](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html) | Still photos (phone, DSLR, mirrorless), locked exposure, ~60% overlap | Splat USDZ only. No mesh; collision proxies by hand; floor not guaranteed at z=0 |
| [Stereo camera](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_stereo.html) | ROS bag with stereo RGB pairs, `camera_info`, `/tf` | Splat USDZ **plus** aligned collision mesh and 2D occupancy map (sim-ready) |

Lidar is optional in both; NVIDIA recommends it for metric scale and pose quality. The stereo workflow runs in the Isaac ROS 4.0 container (CUDA 13; driver 580 is enough). Training uses the open-source [3DGRUT](https://github.com/nv-tlabs/3dgrut) (3DGUT method, Blackwell supported on CUDA 12.8). For Isaac Sim 6.x export the `ParticleField` USD; the older NuRec USDZ volume format is deprecated.

## What a sim-ready scene contains

The sample `nova_carter-wormhole/particle_sh_optimized.usdz` (from [`nvidia/PhysicalAI-Robotics-NuRec`](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec)) is one USD stage, Z-up, metres, floor at z=0:

```
/World/gaussians/NuRec/.../background_gaussians   ParticleField3DGaussianSplat (the splat)
/World/mesh                                       Mesh with PhysicsCollisionAPI + PhysicsMeshCollisionAPI
/World/rig_trajectories                           capture cameras (render quality is best near this path)
occupancy_map.{png,yaml}                          ROS-style occupancy grid, 2 cm resolution
```

Variants: `particle_spg-runtime.usdz` applies PPISP (learned camera ISP) through an Isaac Sim render graph and needs `--enable omni.rtx.spg` plus the carb settings in `isaacsim.replicator.nurec_utils/config/nurec_config.yaml`; `volume.usdz` is the legacy format.

## Use a NuRec scene in Arena

1. **Download.** The dataset is gated: accept the terms on Hugging Face, `hf auth login`, then `bash scripts/download.sh` (the entry pulls one scene, ~3.7 GB, and skips the 18 GB raw images).
2. **Register it as a background.** `arena_nurec/nurec_wormhole_env.py` subclasses `LibraryBackground` with `usd_path` pointing at the USDZ and `object_min_z` for the drop check, and defines an environment (`cube_goal_pose`'s Franka + cube) that uses it. Arena loads the module through its `--external_environment_class_path` hook, so the submodule is not modified.
3. **Run.** `bash sim/nurec/arena_smoke_test.sh` runs 60 zero-action steps headless and writes per-camera videos to `eval/system2/IsaacLab-Arena/outputs/<timestamp>/`. Add `--viz kit` for a GUI and fly the viewport camera to walk through the scene.

To add your own scene: copy the background class, point `usd_path` at your USDZ, and add a `scripts/download.sh` entry if it is hosted. For the LLM environment generator (`scripts/arena_envgen.sh`), the background then appears in the `BACKGROUNDS` catalog under its registered name.

## Reconstruct a scene with 3DGRUT

[3DGRUT](https://github.com/nv-tlabs/3dgrut) is the submodule at `3dgrut/`. Install once (CUDA 12.8 is downloaded into its venv; nothing system-wide):

```bash
cd sim/nurec/3dgrut
env -u DISPLAY FORCE_LOCAL_CUDA=1 CUDA_VERSION=12 ./scripts/create_venv_cuda.sh 3dgrut   # ~4 GB CUDA download, cached in /tmp
source .venv/bin/activate && env -u DISPLAY ./install_env_uv.sh
```

`DISPLAY` must be unset: the CUDA runfile is a makeself archive that tries to spawn an `xterm` when it sees a display but no TTY, and fails with `exec: -title: not found`.

Then train on a COLMAP dataset (`sparse/0/*.bin` + `images/`) and export a ParticleField USD:

```bash
bash sim/nurec/train.sh <colmap_dir> <name> [hydra overrides...]     # 3DGUT + MCMC, every 8th image held out, USD export
bash sim/nurec/train.sh datasets/nurec-zh_lounge/zh_lounge/colmap zh_lounge
```

Runs land in `sim/nurec/runs/<name>/` (gitignored). The `zh_lounge` entry in `scripts/download.sh` is the test case: NVIDIA's 374 photos + COLMAP poses of an office lounge, plus their own reconstruction in `usd/` to compare against.

**Results (2026-10-07, RTX 5090 Laptop, `zh_lounge`, 47 held-out views, 1M Gaussians = the MCMC cap):**

| Run | Iterations | Wall time | PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| `zh_lounge_fast` (`n_iterations=7000`) | 7k | ~5 min incl. first-run kernel compile | 24.9 dB | 0.89 | 0.39 |
| `zh_lounge` (default) | 30k | 31 min | 27.2 dB | 0.92 | 0.34 |

Both export a 236 MB `export_last_lightfield.usdz` with one `ParticleField3DGaussianSplat` prim; both load in Arena (`--background nurec_zh_lounge_ours`).

The export is **Y-up** (COLMAP convention) with no authored extent, while Isaac Sim stages are Z-up; NVIDIA's reference `usd/zh_lounge.usda` applies a rotation to level it. In Arena, give the background an `initial_pose` that rotates +90° about X (quaternion xyzw `(0.7071, 0, 0, 0.7071)`), then nudge the height. The mono workflow gives no collision mesh, so add proxies by hand.

## Reconstruct your own space

Quick test (mono): 200–400 photos of one area with a phone, locked focus/exposure, covering robot camera heights (~1 m looking forward, 30–60 cm above work surfaces looking down); COLMAP for poses; `3dgrut train.py --config-name apps/colmap_3dgut_mcmc.yaml ... export_usd.enabled=true`; load in Isaac Sim; hand-place collision boxes for desk tops and the floor.

Full pipeline (stereo): record a ROS bag from a stereo RGB camera (ZED 2i, or the robot's own cameras) with poses; run the NuRec stereo workflow container; it produces the splat, mesh and occupancy map together.

## Gotchas (learned on this host)

- **Launch flags:** NuRec needs `--/renderer/multiGpu/enabled=false` (passed via Arena's `--kit_args`). The plain particle stage needs nothing else.
- **No `--headless` flag** in this Arena build; headless is the default, set `HEADLESS=1` in the env if needed.
- **Viewport video is broken** on this Isaac Lab branch (`rgb_array` render returns no frames). Use `--record_camera_video`; it flushes at episode reset, so keep `episode_length_s` shorter than the run (the sample env uses 3 s).
- **`configclass`** is imported from `isaaclab.utils.configclass`, not `isaaclab.utils`.
- **Robot placement:** the sample room's origin is under a table; put robots on open floor.
- **Lighting is baked** into the splat. Inserted objects are lit by the RTX lights you add and cast no shadows onto the splat. Nothing in the splat can move; articulated things need separate assets.
- **Quality follows the capture path.** Ground-robot captures look right at 0.3–1.2 m; standing-height views show floaters.
- **RoboDojo (Isaac Sim 5.1)** should load the `v0.0`-tagged assets; untested. **RoboTwin (SAPIEN)** cannot render splats.
