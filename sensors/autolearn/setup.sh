#!/usr/bin/env bash
# Set up sensors/autolearn on this machine, step by step and idempotently. Everything lands in this folder (the
# gitignored state dirs) or in the two submodules' own venvs; the Docker image is the one exception.
#
#   setup.sh geniesim    the Genie Sim CLI venv, asset pack, image, container, RT Engine workspace, crash-reporter hook
#   setup.sh aspire      ASPIRE's LIBERO venv, SAM3 + Contact-GraspNet + PyRoki venvs, the adapter's extra packages, the LeRobot venv
#   setup.sh behavior    BEHAVIOR-1K: CUDA 12.8 toolkit, Isaac Sim 5.1 venv, the datasets (~40 GB, non-commercial terms)
#   setup.sh services    start ASPIRE's SAM3 and Contact-GraspNet (ports 8114-8115) in the background
#   setup.sh benchmark [CONFIG]   start a Genie Sim benchmark run in the container against the adapter on port 8999
#   setup.sh all         geniesim, aspire, behavior
#
# Needs: uv, docker with the NVIDIA container toolkit, a Hugging Face login with access to facebook/sam3
# (`uvx hf auth login`), and for behavior the CUDA 12.8 runfile at $CUDA_RUNFILE (default /tmp/cuda_12.8.1_linux.run).
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

case "${1:-}" in
  geniesim|aspire|behavior|services) "$1" ;;
  benchmark) benchmark "${2:-}" ;;
  all) geniesim; aspire; behavior ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
