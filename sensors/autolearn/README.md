# sensors/autolearn

Robot skills learned by coding agents, then distilled into a VLA: a real site becomes a [Genie Sim](https://github.com/AgibotTech/genie_sim) scene with tasks, [ASPIRE](https://github.com/NVlabs/ASPIRE) agents learn programs that solve them, the programs collect demonstrations, and the flywheel trains and evaluates a VLA on them. Evaluation failures become the next campaign's tasks.

```mermaid
flowchart LR
  S["real site<br/>(panorama)"] --> G["Genie Sim<br/>scene + task variants"]
  G --> A["ASPIRE campaign<br/>programs + verified skills"]
  A --> D["run programs on seeds<br/>successes → LeRobot → XPolicyLab"]
  D --> T["train.sbatch<br/>VLA → MLflow"]
  T --> E["eval: Genie Sim benchmark,<br/>real robot"]
  E -.failed tasks, layouts.-> G
  E -.real episodes.-> D
```

## Layout

Code lives here in git; everything it produces or caches lives in the flywheel, under `$STATE_DIR/autolearn/` (`/tier1/mistral/flywheel/autolearn/`) or in the S3 buckets, never in this checkout. The two upstream submodules stay as published: anything they need changed is wrapped beside them.

| In git (`sensors/autolearn/`) | Contents |
|---|---|
| `aspire/` | [ASPIRE](https://github.com/NVlabs/ASPIRE) (upstream submodule, untouched): coding agents write and debug robot programs; LIBERO-Pro, Robosuite, BEHAVIOR-1K |
| `genie_sim/` | [Genie Sim 3.0](https://github.com/AgibotTech/genie_sim) (upstream submodule, untouched, MPL-2.0): Isaac Sim 5.1 scenes from a panorama, task generation, benchmark, data collection |
| `gb300/` | what runs Genie Sim on the GB300 nodes: `render_bench.py` + `render_bench.sbatch` (rendering speed); planned: `build.sh` (upstream's `Dockerfile.5.1` built for arm64 without the x86 `ik_solver` wheel), `up.sh` (the container on a Slurm GPU, caches on `/tier1`, ROS on domain 42), `ik_shim/` (a drop-in `ik_solver` package that calls `ik_server/` over RPC, [issue 149](https://github.com/AgibotTech/genie_sim/issues/149)) |
| `cap-harness-sandboxed.sh` | ASPIRE's `cap-harness` inside bubblewrap with the user's credentials hidden; what the agent loops run programs through |
| `hooks/sitecustomize.py` | turns Isaac Sim's crash reporter off for Genie Sim's engine, which starts `SimulationApp` with no settings: copied into the container and put on `PYTHONPATH` |
| planned: `adapter/` | the Genie Sim adapter for ASPIRE (plan stage 2) and the corobot bridge for XPolicyLab's policy server (stage 5) |
| planned: [`services/gitea/autolearn/`](../../services/gitea/README.md) | the Gitea workflows: scene building (stage 1), `geniesim_to_xpolicylab` (stage 3), `autolearn-cycle` (stage 6), beside the other stages' workflows |

| On the cluster (not in git) | Contents |
|---|---|
| `$STATE_DIR/autolearn/isaac-cache/` | Isaac Sim's kit, shader and Omniverse caches (the first run compiles shaders for minutes; later runs reuse them) |
| `$STATE_DIR/autolearn/logs/`, `results/` | Slurm job logs and the benchmarks' JSON |
| `$STATE_DIR/autolearn/venv-geniesim/` | the Genie Sim CLI venv (`geniesim docker5.1 ...`) |
| `$STATE_DIR/autolearn/campaigns/<campaign>/` | ASPIRE's outputs (`--output-root`): programs, runs, verified skills; scores go to MLflow |
| `$STATE_DIR/hf-home/` | Hugging Face cache shared with the stack (SAM3 weights, ...) |
| `s3://raw/open_datasets/GenieSimAssets/` | Genie Sim's asset pack (29 GB, a pip package), fetched by `download-datasets-hf`; mounted into the container as its assets |
| `s3://raw/internal_datasets/scenes/<site>/` | panoramas of real sites (stage 1) |
| `s3://processed/xpolicylab/geniesim/<task>/` | collected demonstrations after `geniesim_to_xpolicylab` (stage 3), where [`train.sbatch`](../../services/slurm/README.md) reads them |
| MLflow | campaign scores (experiments), VLA checkpoints (registry), evaluation results |

On this laptop (x86_64, no `/tier1`), everything ASPIRE or Genie Sim produces stays inside `sensors/autolearn/`: `isaac-cache/`, `.venv-geniesim/` and the asset pack `GenieSimAssets/` (`.gitignore`d here), ASPIRE's venvs and campaigns where its scripts put them (`aspire/.venv-*`, `aspire/outputs/`, ignored by its git), the ROS 2 workspace build in `genie_sim/devel*`; the Hugging Face cache is the user's `~/.cache/huggingface`.

## Plan

| Stage | Exists | To build |
|---|---|---|
| 1. Real site → scene: panorama into `raw/internal_datasets/scenes/<site>/`; `geniesim_world` (PanoRecon) builds the 3D scene, `geniesim_generator` places objects and writes task variants with success checks | S3, FiftyOne sync; LLM gateway for the generator (`LLM_URL`) | a Gitea workflow on an RTX node; embeddings for the generator's asset search |
| 2. Learn: ASPIRE fix loop per task, in parallel on the GB300; scores to MLflow | ASPIRE on LIBERO (below) | **a Genie Sim adapter for ASPIRE** (`reset`, `step`, observations, success over ROS 2 or the benchmark's inference socket, via `run_program_with_api`): the piece everything else waits on |
| 3. Collect: run verified programs on many seeds, keep successes; `geniesim dataset convert agibot-to-lerobot`; LeRobot → XPolicyLab | [`galaxeaOpenWorldDataset_to_xpolicylab`](../../services/gitea/adapter/README.md) reads LeRobot v2.1 | a `geniesim_to_xpolicylab` adapter workflow (a copy) |
| 4. Train: `train.sbatch <vla> <embodiment> geniesim/<task>` | [slurm](../../services/slurm/README.md) for ACT, DP, MLP | a `models/<vla>/` recipe (XPolicyLab ships `Pi_05`, `GR00T_N17`, `OpenVLA_OFT`, `GalaxeaVLA`) and an embodiment YAML |
| 5. Evaluate: `geniesim benchmark batch` against the served VLA; XPolicyLab real-robot eval; both to MLflow | MLflow, `evaluate.sbatch` | a bridge from XPolicyLab's policy server to Genie Sim's corobot WebSocket (`{"method": "infer"}` → `left_arm`, `right_arm`, `waist`, `head`) |
| 6. Loop: a scheduled workflow reads MLflow, turns failed tasks and layouts (sim and real) into the next campaign's variants and seeds, adds real successes to the training mix, retrains | Gitea Actions, cron | the `autolearn-cycle` workflow |

Models, and where they run:

| Model | Used by | Runs on |
|---|---|---|
| `glm-5.3-flash`, `qwen3.5-122b-a10b` (LLM gateway) | Genie Sim's generator and LLM eval judge | Dynamo (served) |
| `moge-3-vitg`, `llmdet_large`, `sam2.1-hiera-large`, `Qwen-Image-Edit-2511`, `TripoSG` | own assets and scene edits | Triton (served) |
| ml-sharp, DA360 | `geniesim_world` PanoRecon | a batch job on an RTX node |
| Coding-agent LLM | ASPIRE's workers, verifiers, coordinator | Claude Code, or an open coding agent on the LLM gateway |
| SAM3 (gated), Contact-GraspNet, PyRoki, cuRobo | ASPIRE's programs (ports 8114–8116, ASPIRE's own HTTP protocol) | ASPIRE's service stack per node; Triton only with new backends and a client shim |
| The VLA | collection-time eval, deployment | served from MLflow by a Slurm job, behind the corobot bridge |

Decisions to take first: the embodiment (Genie Sim ships the AgiBot G2, Franka, UR5, ARX and Agilex; a Galaxea R1 Pro must be added with Genie Sim's `add-robot` skill), and the licences of SAM3, Contact-GraspNet's PyTorch port, ASPIRE and Genie Sim's assets for derived data.

## Where things run

A GB300 GPU renders fast enough for Genie Sim (`gb300/render_bench.sbatch`, Isaac Sim 6.1 from pip, Genie Sim's real-time path tracer without DLSS or the DL denoiser, which need NGX the GB300 lacks): expect ~40 steps/s with one 640x480 camera and ~16 steps/s with three 1280x720 cameras, frames fetched each step, on Isaac's full warehouse; the offline path tracer (16 spp, OptiX-denoised) is the slow one (~150 steps in 3 min). So scene building, data collection and benchmarks can run on the GB300 too, beside ASPIRE's agent loops, its perception services, LIBERO (MuJoCo), the LLMs, VLA training and serving; a Genie Sim scene's materials and the grain without the DL denoiser are still to check. `scene_reconstruction` (CUDA 11.8, Ada only) doesn't run on the GB300 at all.

## Genie Sim

`A` is the state directory (`/tier1/mistral/flywheel/autolearn` on the cluster, `sensors/autolearn` here) and `ASSETS` the asset pack (`raw/open_datasets/GenieSimAssets` from `download-datasets-hf` there, `$A/GenieSimAssets` from `uvx hf download agibot-world/GenieSimAssets --repo-type dataset --local-dir $A/GenieSimAssets` here). The pack is [CC BY-NC-SA 4.0](https://huggingface.co/datasets/agibot-world/GenieSimAssets): scenes, demonstrations and models derived from it are non-commercial and share-alike; use your own assets for anything else.

1. Install the CLI with the asset pack as its `geniesim_assets` package.
   ```bash
   uv venv --python 3.12 $A/.venv-geniesim && VIRTUAL_ENV=$A/.venv-geniesim uv pip install pip -e sensors/autolearn/genie_sim/source/geniesim_cli -e $ASSETS
   ```
2. Build the image (Isaac Sim 5.1 base; ~25 min, 26 GB) and start it headless. The container runs with host networking and `--privileged`; its entrypoint keeps ROS 2 on localhost, and the launch below adds domain 42. Isaac's caches go to `$A/isaac-cache`.
   ```bash
   cd sensors/autolearn/genie_sim && GENIESIM_WIP_ACK=1 $A/.venv-geniesim/bin/geniesim docker5.1 build
   GENIESIM_CACHE_ROOT=$A/isaac-cache $A/.venv-geniesim/bin/geniesim docker5.1 up --headless
   ```
3. Build the RT Engine's ROS 2 workspace (~3 min, into `genie_sim/devel`) and install the crash-reporter hook.
   ```bash
   docker exec -u $(id -u):$(id -g) geniesim3 bash -lc 'cd /workspace && geniesim ros build dev'
   docker exec geniesim3 mkdir -p /home/isaac-sim/.local/hooks && docker cp ../hooks/sitecustomize.py geniesim3:/home/isaac-sim/.local/hooks/
   ```
4. Launch a scene headless (the G2 pick-and-place on PhysX) and measure its camera rate.
   ```bash
   docker exec -u $(id -u):$(id -g) -e ROS_DOMAIN_ID=42 -e PYTHONPATH=/home/isaac-sim/.local/hooks -e ISAACSIM_HEADLESS=1 geniesim3 bash -lc \
     'source /opt/ros/jazzy/setup.bash && source /workspace/devel/setup.bash && cd /workspace && ros2 launch genie_sim_bringup app.launch.py scene:=scene_pnp_g2_op launcher_config:=launcher_ovrtx_isaac_physx headless:=true'
   docker exec -u $(id -u):$(id -g) -e ROS_DOMAIN_ID=42 geniesim3 bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 topic hz /genie_sim/head_front_camera_rgb/image_raw --window 30'
   ```

On this laptop: `Isaac Sim physics engine ready` after ~6 min cold (shader compile; tens of seconds warm), physics at 100 Hz, the render node at its 30 fps target with two cameras (`Render Stats` in the log: ~27 ms a frame, no drops), the head camera topic at ~26 Hz, 5.6 GB of GPU memory. Stop a scene with `docker exec geniesim3 pkill -INT -f "ros2 launch"`; `geniesim docker5.1 down` removes the container (the caches and the workspace build stay).

### On the GB300

Isaac Sim's 5.1.0 image is published for arm64, so the build is possible. Rendering speed is measured without Genie Sim's image (its `ik_solver` wheel is x86-only, [issue 149](https://github.com/AgibotTech/genie_sim/issues/149)): `gb300/render_bench.sbatch` runs `gb300/render_bench.py` on a Slurm-allocated GPU with Genie Sim's renderer settings and prints steps per second for 1 and 3 cameras; caches, logs and results go to `$A`. Run it with a pip Isaac Sim 6.x (IsaacLab-Arena's venv): in NVIDIA's `isaac-sim:5.1.0` arm64 image the RTX renderer never starts (`Failed to acquire exclusive lock to data store`, then `Failed to initialize rtx::shaderdb::ContextManager`, with caches on `/tier1`, on local disk or with host IPC alike), which Genie Sim's image inherits for its `omni_python`.
```bash
ISAAC_VENV=$PWD/eval/system2/IsaacLab-Arena/.venv sbatch --export=ALL -w <GB300 node> sensors/autolearn/gb300/render_bench.sbatch
```

Decide by steps per second: under about 5 the RTX nodes keep the rendering; above that the GB300 can take data collection too, and the steps above apply with `A=/tier1/mistral/flywheel/autolearn` (expect to patch `genie_sim/docker/Dockerfile.5.1` for the arm64 Isaac Sim wheels, and start the container from a Slurm GPU job with `GENIESIM_ASSETS_SRC` pointing at the bucket's pack). A render failing with `NGX CreateFeature failed` needs the no-DLSS renderer (FXAA, no DL denoiser, OptiX path tracing), as IsaacLab-Arena needed there (`git show main:sim/isaaclab_arena/hooks/sitecustomize.py`).

Notes:
- Scenes and launchers are YAMLs in `genie_sim/source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config/` (`scene_flat_fr3` is a Franka; `scene_wbc_g2_sp` whole-body control on the Newton solver). `geniesim benchmark run` needs a policy server (corobot WebSocket); `geniesim autocollect` (scripted collection) runs its own container from `source/data_collection/`.

## ASPIRE on LIBERO

1. Initialise the submodule and set the paths (every step below assumes them).
   ```bash
   git submodule update --init sensors/autolearn/aspire
   A=$PWD/sensors/autolearn && cd $A/aspire
   export LIBERO_CONFIG_PATH=$PWD/.libero TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
   ```
2. Log in to Hugging Face with access to the gated [SAM3](https://huggingface.co/facebook/sam3) weights: `uvx hf auth login`.
3. Install LIBERO and the model services (SAM3, Contact-GraspNet, PyRoki), then start the services on ports 8114–8116.
   ```bash
   scripts/bootstrap_libero.sh
   scripts/bootstrap_providers.sh --providers sam3,contact_graspnet,pyroki --profile rtx5090
   scripts/supervise_services.sh --profile rtx5090 --providers sam3,contact_graspnet,pyroki &
   ```
4. Smoke test: the example program for "pick up the alphabet soup and place it in the basket"; expect `task_success=True`.
   ```bash
   ../cap-harness-sandboxed.sh run --benchmark libero-pro --suite libero_object_swap --task-id 0 --seed 1 --init-mode seeded \
     --program examples/libero/object_swap_seed1.py --output-root outputs/examples --max-steps 2000
   ```
5. Run a campaign: ASPIRE's `iterative-debugging` skill (`aspire/.claude/skills/iterative-debugging/SKILL.md`) with a coding agent as coordinator, `CAP_HARNESS=$A/cap-harness-sandboxed.sh`, and ASPIRE's helper scripts run with `.venv-libero/bin/python`. Expect a task to take about 15 min of Stage 1 (5 dev seeds), 15 min of held-out seeds and 5 min of skill verification on this laptop; `libero_goal_swap` task 0 (open the middle drawer) reached 10/10 held-out.

Notes:
- Pass `--init-mode seeded`: ASPIRE v1.0 crashes resetting LIBERO in the default `saved` mode (`adapter.py` tests a NumPy array with `not`), which also fails `cap-harness doctor --reset` and `validate --plan libero-smoke`.
- Verified skills belong in `aspire/.claude/skills/iterative-debugging/skills/`, a tracked file of the submodule: promote them on a fork, or keep them in the campaign's `outputs/runs/<campaign>/`.
- ASPIRE runs model-generated Python with full access to the machine; the wrapper hides credentials from the programs, not from the agent. Run agent loops on a host without credentials.
- BEHAVIOR-1K (Galaxea R1 Pro) pickups can't run on RTX 50-series GPUs: OmniGibson 3.9.2 drops the cuRobo R1 Pro embodiment ASPIRE plans with.
