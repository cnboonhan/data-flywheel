# sensors/autolearn

[ASPIRE](https://github.com/NVlabs/ASPIRE) (`aspire/`, upstream submodule): coding agents that write and refine robot skills, on LIBERO-Pro and Robosuite (Franka) and BEHAVIOR-1K (Galaxea R1 Pro, Isaac Sim 5.1). Everything it downloads stays in this folder (gitignored): `behavior-data/` (BEHAVIOR datasets, ~37 GB), `cuda-12.8/` (toolkit for cuRobo), `hf-cache/` (SAM3 weights), and ASPIRE's venvs and outputs inside `aspire/`.

## Set up

1. Initialise the submodule and set the paths (every step below assumes them).
   ```bash
   git submodule update --init sensors/autolearn/aspire
   A=$PWD/sensors/autolearn
   export HF_HOME=$A/hf-cache HF_TOKEN_PATH=~/.cache/huggingface/token
   export CAP_HARNESS_BEHAVIOR_DATA=$A/behavior-data CAP_HARNESS_CUDA_HOME=$A/cuda-12.8
   ```
2. Log in to Hugging Face with access to the gated [SAM3](https://huggingface.co/facebook/sam3) weights: `uvx hf auth login`.
3. Install a CUDA 12.x toolkit into `$A/cuda-12.8` (no root; cuRobo compiles against it):
   ```bash
   sh cuda_12.8.1_linux.run --silent --toolkit --toolkitpath=$A/cuda-12.8 --no-man-page --no-drm --override
   ```
4. Install LIBERO and the model services, and start the services (SAM3, Contact-GraspNet, PyRoki on ports 8114–8116); profiles are in `aspire/configs/profiles/`.
   ```bash
   cd $A/aspire
   scripts/bootstrap_libero.sh
   scripts/bootstrap_providers.sh --providers sam3,contact_graspnet,pyroki --profile rtx5090
   scripts/supervise_services.sh --profile rtx5090 --providers sam3,contact_graspnet,pyroki &
   ```
5. For BEHAVIOR-1K, install Isaac Sim 5.1, cuRobo and the datasets. `--accept-dataset-tos` accepts the BEHAVIOR Data Bundle terms (non-commercial academic research). If its import check fails with `pyarrow requires NumPy 2.0`, install `pyarrow<23` into the venv and rerun.
   ```bash
   scripts/bootstrap_behavior.sh --accept-dataset-tos --profile rtx5090
   uv pip install --python .venv-behavior "pyarrow<23"     # only after that failure
   ```

## Smoke test

```bash
cd $A/aspire
export LIBERO_CONFIG_PATH=$PWD/.libero TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
.venv-libero/bin/cap-harness run --benchmark libero-pro --suite libero_object_swap --task-id 0 --seed 1 --init-mode seeded \
  --program examples/libero/object_swap_seed1.py --output-root outputs/examples --max-steps 2000
OMNIGIBSON_DATA_PATH=$CAP_HARNESS_BEHAVIOR_DATA scripts/validate_behavior.sh structural   # reset both R1 Pro tasks
OMNIGIBSON_DATA_PATH=$CAP_HARNESS_BEHAVIOR_DATA scripts/validate_behavior.sh pickup       # radio and soda-can pickups
```

Expect `task_success=True` for the LIBERO run, 6/6 structural, and about 2/3 per pickup task.

Notes:
- LIBERO's default `--init-mode saved` crashes on reset in ASPIRE v1.0 (`adapter.py` tests a NumPy array with `not`), so `cap-harness doctor --reset` and `validate --plan libero-smoke` fail; use `--init-mode seeded` for runs.
- BEHAVIOR pickups need a GPU other than sm_120 (RTX 50-series): OmniGibson 3.9.2 drops cuRobo's DEFAULT R1 Pro embodiment there, and every arm plan fails with `KeyError: CuRoboEmbodimentSelection.DEFAULT`. Structural resets still pass.
- Isaac Sim's exit after `validate_behavior.sh` can segfault; the summary is already written by then.
- ASPIRE runs model-generated Python with full access: run its agent loops on a host without credentials.
