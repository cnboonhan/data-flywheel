#!/usr/bin/env bash
# Set up sensors/autolearn on this machine, step by step and idempotently. Everything lands in this folder (the
# gitignored state dirs) or in the two submodules' own venvs; the Docker image is the one exception. On arm64 (the
# GB300 nodes) everything lands under $STATE on shared storage instead, and only BEHAVIOR-1K collection is set up
# (see "arm64" below).
#
#   setup.sh geniesim    the Genie Sim CLI venv, asset pack, image, container, RT Engine workspace, crash-reporter hook
#   setup.sh aspire      ASPIRE's LIBERO venv, SAM3 + Contact-GraspNet + PyRoki venvs, the adapter's extra packages, the LeRobot venv
#   setup.sh behavior    BEHAVIOR-1K: CUDA 12.8 toolkit, Isaac Sim 5.1 venv, the datasets (~40 GB, non-commercial terms)
#   setup.sh services    start ASPIRE's SAM3 and Contact-GraspNet (ports 8114-8115) in the background
#   setup.sh benchmark [CONFIG]   start a Genie Sim benchmark run in the container against the adapter on port 8999
#   setup.sh all         geniesim, aspire, behavior
#
# Needs: uv, docker with the NVIDIA container toolkit, a Hugging Face login with access to facebook/sam3
# (`uvx hf auth login`, or HF_TOKEN), and for behavior the CUDA 12.8 runfile at $CUDA_RUNFILE (default
# /tmp/cuda_12.8.1_linux.run; not on arm64).
set -euo pipefail
A="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${CAP_HARNESS_PROFILE:-rtx5090}"
export HF_HOME="$A/hf-cache" HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}"
log() { printf '\n== %s\n' "$*"; }

geniesim() {
  cd "$A"
  git -C "$A/.." submodule update --init autolearn/genie_sim
  if [[ ! -x .venv-geniesim/bin/geniesim ]]; then
    log "Genie Sim CLI venv"
    uv venv --python 3.12 .venv-geniesim
    VIRTUAL_ENV="$A/.venv-geniesim" uv pip install pip -e genie_sim/source/geniesim_cli
  fi
  if [[ ! -f GenieSimAssets/pyproject.toml ]]; then
    log "Genie Sim asset pack (29 GB, CC BY-NC-SA 4.0)"
    uvx hf download agibot-world/GenieSimAssets --repo-type dataset --local-dir GenieSimAssets
  fi
  .venv-geniesim/bin/python -c 'import geniesim_assets' 2>/dev/null || VIRTUAL_ENV="$A/.venv-geniesim" uv pip install -e GenieSimAssets
  cd genie_sim
  if ! docker image inspect registry.agibot.com/genie-sim/geniesim3:latest >/dev/null 2>&1; then
    log "Genie Sim image (Isaac Sim 5.1 base, ~25 min)"
    GENIESIM_WIP_ACK=1 ../.venv-geniesim/bin/geniesim docker5.1 build
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx geniesim3; then
    log "Genie Sim container (headless, caches in isaac-cache/)"
    GENIESIM_CACHE_ROOT="$A/isaac-cache" ../.venv-geniesim/bin/geniesim docker5.1 up --headless
  fi
  if [[ ! -f devel/setup.bash ]]; then
    log "RT Engine ROS 2 workspace (~3 min)"
    docker exec -u "$(id -u):$(id -g)" geniesim3 bash -lc 'cd /workspace && geniesim ros build dev'
  fi
  log "crash-reporter hook into the container"
  docker exec geniesim3 mkdir -p /home/isaac-sim/.local/hooks
  docker cp "$A/hooks/sitecustomize.py" geniesim3:/home/isaac-sim/.local/hooks/sitecustomize.py
  docker exec -u "$(id -u):$(id -g)" -e PYTHONPATH=/home/isaac-sim/.local/hooks geniesim3 \
    python3 -c 'import isaacsim; assert not isaacsim.SimulationApp.DEFAULT_LAUNCHER_CONFIG["enable_crashreporter"]; print("hook ok")'
}

aspire() {
  cd "$A"
  git -C "$A/.." submodule update --init autolearn/aspire
  if [[ ! -d hf-cache/hub/models--facebook--sam3 ]]; then
    log "SAM3 weights into hf-cache/ (gated: needs access on Hugging Face)"
    uvx hf download facebook/sam3
  fi
  cd aspire
  log "LIBERO venv"
  scripts/bootstrap_libero.sh
  log "SAM3, Contact-GraspNet, PyRoki venvs"
  scripts/bootstrap_providers.sh --providers sam3,contact_graspnet,pyroki --profile "$PROFILE"
  log "adapter packages into the PyRoki venv"
  VIRTUAL_ENV="$A/aspire/.venv-pyroki" uv pip install msgpack opencv-python-headless
  log "LeRobot venv for adapter/to_lerobot.py (0.3.3: the last release writing dataset v2.1, which the flywheel reads)"
  [[ -d .venv-lerobot ]] || uv venv --python 3.10 .venv-lerobot
  uv pip install --python .venv-lerobot/bin/python "lerobot==0.3.3" h5py pyarrow pandas opencv-python-headless av boto3   # + the xspark adapter's needs
}

behavior() {
  cd "$A"
  if [[ ! -x cuda-12.8/bin/nvcc ]]; then
    log "CUDA 12.8 toolkit into cuda-12.8/ (cuRobo compiles against it)"
    env -u DISPLAY sh "${CUDA_RUNFILE:-/tmp/cuda_12.8.1_linux.run}" --silent --toolkit --toolkitpath="$A/cuda-12.8" --no-man-page --no-drm --override
  fi
  cd aspire
  git submodule update --init third_party/BEHAVIOR-1K
  export CAP_HARNESS_CUDA_HOME="$A/cuda-12.8" CAP_HARNESS_BEHAVIOR_DATA="$A/behavior-data"
  log "BEHAVIOR venv (Isaac Sim 5.1, cuRobo) and datasets; accepts the BEHAVIOR Data Bundle terms (non-commercial)"
  # ASPIRE's bootstrap leaves pyarrow unpinned and its import check then fails on NumPy 1.x; pin it and rerun
  scripts/bootstrap_behavior.sh --accept-dataset-tos --profile "$PROFILE" || true
  VIRTUAL_ENV="$A/aspire/.venv-behavior" uv pip install "pyarrow<23" "numpy<2"
  scripts/bootstrap_behavior.sh --accept-dataset-tos --profile "$PROFILE"
}

services() {
  cd "$A/aspire"
  pgrep -f 'cap_harness.providers.sam3' >/dev/null && { echo "services already running"; return; }
  log "SAM3 (8114) and Contact-GraspNet (8115)"
  nohup scripts/supervise_services.sh --profile "$PROFILE" --providers sam3,contact_graspnet > "$A/aspire/outputs/services.log" 2>&1 &
  for _ in $(seq 60); do grep -q 'all provider services ready' "$A/aspire/outputs/services.log" 2>/dev/null && break; sleep 2; done
  tail -2 "$A/aspire/outputs/services.log"
}

benchmark() {
  local config="${1:-g2op_if_pick_block_color}"
  docker exec -u "$(id -u):$(id -g)" -e PYTHONPATH=/home/isaac-sim/.local/hooks -e ROS_DOMAIN_ID=42 geniesim3 bash -lc \
    "cd /workspace && geniesim benchmark run $config --infer-host=127.0.0.1:8999 --app.headless=true --benchmark.num_episode=1"
}

# ---- arm64 (GB300 nodes): BEHAVIOR-1K collection only, in Slurm jobs (gb300/collect_behavior.sbatch) ----
# Differences from the steps above, all forced by arm64 or the cluster:
#   - state under $STATE (shared storage, group-readable), not in this checkout; gb300/behavior.env points jobs at it
#   - no cuRobo: run_behavior.py uses PyRoki (adapter/ik.py), and OmniGibson's cuRobo would need compiling for sm_103
#   - torch 2.7 cu128 and Isaac Sim 5.1 have arm64 wheels; libGLU and libOpenGL are missing on the nodes and are
#     unpacked from Ubuntu's arm64 packages into $STATE/libs (no system change)
#   - SAM3: torch 2.9.1 cu130 (has sm_103); decord has no arm64 wheels, and SAM3 imports it at module level though the
#     service only segments images, so a stand-in that refuses video takes its place
#   - no bubblewrap on the nodes: gb300/run_behavior.sh runs reviewed policies unsandboxed; no agent campaigns here
#   - not set up: Genie Sim (x86-only image and ik_solver), LIBERO, Contact-GraspNet
STATE="${AUTOLEARN_STATE:-/tier1/mistral/flywheel/autolearn}"
B1K_COMMIT=b1979916ec1549b10a4e65e630bc6504a9af1b00     # aspire/configs/dependency-lock.json (BEHAVIOR-1K v3.9.2)
PYROKI_COMMIT=388e43e1fc0d0ee382968d3dd72970fd62a0450c  # same lock
SAM3_COMMIT=6fe87d64a5beb9084923d7a9e002741178635b09    # same lock (Max-Fu/sam3)

src_at() {   # src_at <repo url> <dir> <commit>
  [[ $(git -C "$2" rev-parse HEAD 2>/dev/null) == "$3" ]] && return
  [[ -d $2/.git ]] || git clone -q --filter=blob:none "$1" "$2"
  git -C "$2" fetch -q origin "$3" && git -C "$2" checkout -q "$3"
}

arm64_libs() {
  local libs=$STATE/libs/lib tmp
  [[ -e $libs/libGLU.so.1 && -e $libs/libOpenGL.so.0 ]] && return
  log "libGLU and libOpenGL into $libs (the nodes lack them; Isaac's MDL and RTX renderer need them)"
  tmp=$(mktemp -d); (cd "$tmp" && apt-get download libglu1-mesa libopengl0)
  for f in "$tmp"/*.deb; do dpkg-deb -x "$f" "$tmp/root"; done
  mkdir -p "$libs"; cp -a "$tmp"/root/usr/lib/aarch64-linux-gnu/lib{GLU,OpenGL}.so* "$libs/"; rm -rf "$tmp"
}

arm64_behavior() {
  local V=$STATE/venv-behavior B1K=$STATE/src/BEHAVIOR-1K
  local PIP=(uv pip install --python "$V/bin/python")
  arm64_libs
  src_at https://github.com/StanfordVL/BEHAVIOR-1K "$B1K" "$B1K_COMMIT"
  if [[ ! -f $V/.behavior-ok ]]; then
    log "BEHAVIOR venv $V (Isaac Sim 5.1, OmniGibson 3.9.2, PyRoki; no cuRobo)"
    uv venv --allow-existing --python 3.11 "$V"   # Isaac Sim 5.1 wheels are cp311-only
    printf '%s\n' torch==2.7.0 torchvision==0.22.0 'numpy<2' 'setuptools>=71,<81' 'pyarrow<23' > "$V/constraints.txt"
    # pyarrow<23: an unpinned one needs NumPy 2 and breaks OmniGibson's import (as on x86, above)
    export UV_CONSTRAINT=$V/constraints.txt
    "${PIP[@]}" --index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match torch==2.7.0 torchvision==0.22.0
    "${PIP[@]}" 'numpy<2' 'setuptools>=71,<81' wheel ninja psutil 'pyarrow<23'
    "${PIP[@]}" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match 'isaacsim[all,extscache]==5.1.0'
    # bootstrap_behavior.sh's fixups: Isaac's bundled websockets/packaging shadow the venv's; its cffi mismatches OmniGibson's
    "$V/bin/python" -c '
import importlib.util, pathlib, shutil
root = pathlib.Path(importlib.util.find_spec("isaacsim").origin).parent
for pat in ("extscache/**/pip_prebundle/websockets", "extscache/**/pip_prebundle/packaging"):
    for p in root.glob(pat): shutil.rmtree(p, ignore_errors=True)'
    "${PIP[@]}" --reinstall-package cffi cffi==1.17.1
    "${PIP[@]}" --reinstall-package websockets 'websockets>=15.0.1'
    "${PIP[@]}" --editable "$B1K/bddl3"
    "${PIP[@]}" --editable "$B1K/OmniGibson"   # no [primitives]: that extra is cuRobo
    "${PIP[@]}" "pyroki @ git+https://github.com/chungmin99/pyroki.git@$PYROKI_COMMIT" yourdfpy jax   # IK on CPU (behavior.env)
    "${PIP[@]}" numpy requests scipy PyYAML 'imageio[ffmpeg]' pydantic   # ASPIRE's client_dependencies
    "${PIP[@]}" --no-deps --editable "$A/aspire"
    unset UV_CONSTRAINT
    touch "$V/.behavior-ok"
  fi
  log "BEHAVIOR datasets into $STATE/behavior-data (~36 GB; accepts the BEHAVIOR Data Bundle terms: non-commercial)"
  (set -a; . "$A/gb300/behavior.env"; set +a
   mkdir -p "$OMNIGIBSON_DATA_PATH" "$OMNIGIBSON_APPDATA_PATH"   # OmniGibson refuses to import without them
   "$V/bin/python" -c '
import os
from omnigibson.macros import gm
from omnigibson.utils import asset_utils as au
root = gm.DATA_PATH
need = {"robot": os.path.join(root, "omnigibson-robot-assets", "models", "r1pro", "r1pro.yaml"),
        "assets": os.path.join(root, "behavior-1k-assets", "VERSION"),
        "instances": os.path.join(root, "2026-challenge-task-instances", "metadata", "task.jsonl"),
        "key": au.get_key_path()}
if not os.path.exists(need["robot"]): au.download_omnigibson_robot_assets()
if not (os.path.exists(need["assets"]) and os.path.exists(need["key"])): au.download_behavior_1k_assets(accept_license=True)
if not os.path.exists(need["instances"]): au.download_2026_challenge_task_instances()
missing = [k for k, p in need.items() if not os.path.exists(p)]
assert not missing, missing
print("datasets present under", root)')
}

arm64_aspire() {
  local V=$STATE/venv-sam3 L=$STATE/venv-lerobot
  export HF_HOME=$STATE/hf-cache
  if [[ ! -d $HF_HOME/hub/models--facebook--sam3 ]]; then
    log "SAM3 weights into $HF_HOME (gated; SAM License): HF_TOKEN, else services/.env's"
    [[ -n ${HF_TOKEN:-} ]] || HF_TOKEN=$(grep -m1 '^HF_TOKEN=' "$A/../../services/.env" | cut -d= -f2- | tr -d "\"'")
    HF_TOKEN=$HF_TOKEN uvx --from huggingface_hub hf download facebook/sam3
  fi
  src_at https://github.com/Max-Fu/sam3.git "$STATE/src/sam3" "$SAM3_COMMIT"
  if [[ ! -f $V/.sam3-ok ]]; then
    log "SAM3 service venv $V"
    local PIP=(uv pip install --python "$V/bin/python")
    uv venv --allow-existing --python 3.12 "$V"
    "${PIP[@]}" --index-url https://download.pytorch.org/whl/cu130 torch==2.9.1 torchvision==0.24.1   # cu130: has sm_103
    "${PIP[@]}" --no-deps --editable "$STATE/src/sam3"
    # ASPIRE's providers-sam3 extra (aspire/pyproject.toml) without decord, plus SAM3's own runtime imports
    "${PIP[@]}" einops 'psutil>=5.9' 'open_clip_torch>=2.24' 'opencv-python-headless<4.13' 'pillow>=10' 'pycocotools>=2.0' \
      'fastapi>=0.115' 'uvicorn>=0.34' pydantic numpy requests scipy PyYAML huggingface_hub iopath ftfy regex timm hydra-core
    "${PIP[@]}" --no-deps --editable "$A/aspire"
    local d; d=$("$V/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/decord
    mkdir -p "$d"; cat > "$d/__init__.py" <<'PY'
"""Stand-in for decord, which has no arm64 wheels: SAM3 imports it at module level, but ASPIRE's service segments
images only. Reading video through it fails loudly. (sensors/autolearn/setup.sh, arm64)"""
__version__ = "0.0-stub"
def cpu(*a, **k): return None
def gpu(*a, **k): return None
class VideoReader:
    def __init__(self, *a, **k): raise RuntimeError("decord is a stand-in on arm64: no video reading")
class bridge:
    @staticmethod
    def set_bridge(*a, **k): pass
PY
    "$V/bin/python" -c 'from sam3.model_builder import build_sam3_image_model' && touch "$V/.sam3-ok"
  fi
  if [[ ! -x $L/bin/python ]]; then
    log "LeRobot venv $L (0.3.3: the last release writing dataset v2.1)"
    uv venv --python 3.10 "$L"
    uv pip install --python "$L/bin/python" "lerobot==0.3.3" h5py pyarrow pandas opencv-python-headless av boto3
  fi
}

if [[ $(uname -m) == aarch64 ]]; then
  case "${1:-}" in
    aspire) arm64_aspire ;;
    behavior) arm64_behavior ;;
    all) arm64_aspire; arm64_behavior ;;
    *) echo "on arm64 only: setup.sh aspire | behavior | all (BEHAVIOR-1K collection; see the arm64 notes in this file)" >&2; exit 2 ;;
  esac
  exit 0
fi

case "${1:-}" in
  geniesim|aspire|behavior|services) "$1" ;;
  benchmark) benchmark "${2:-}" ;;
  all) geniesim; aspire; behavior ;;
  *) sed -n '2,17p' "$0"; exit 2 ;;
esac
