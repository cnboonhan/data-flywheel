#!/usr/bin/env bash
# Smoke-test a NuRec (Gaussian splat) scene inside IsaacLab-Arena: runs cube_goal_pose's Franka + cube in the
# downloaded nova_carter-wormhole room and records the robot-camera videos under eval/system2/IsaacLab-Arena/outputs/.
#   bash sim/nurec/arena_smoke_test.sh                     # 60 zero-action steps (one 3 s episode + reset), headless
#   bash sim/nurec/arena_smoke_test.sh --num_steps 100     # extra policy_runner args
#   bash sim/nurec/arena_smoke_test.sh -- --background nurec_zh_lounge_ours   # args after -- go to the env (e.g. our own splat)
# Download the scene first with scripts/download.sh (nvidia/PhysicalAI-Robotics-NuRec is gated).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA="${ROOT}/eval/system2/IsaacLab-Arena"
SCENE="${ROOT}/datasets/nurec-nova_carter-wormhole/nova_carter-wormhole/particle_sh_optimized.usdz"
[ -f "${SCENE}" ] || { echo "NuRec scene not found at ${SCENE}; run: bash scripts/download.sh" >&2; exit 1; }

cd "${ARENA}"
export PYTHONPATH="${ROOT}/sim/nurec${PYTHONPATH:+:${PYTHONPATH}}"
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export HEADLESS="${HEADLESS:-1}"   # this Isaac Lab reads headless mode from the env var, not a flag
# NuRec needs multi-GPU rendering off at launch; the plain particle stage needs no other render settings.
# Split "<runner args> -- <env args>": env args must follow the environment name (argparse subcommand).
runner_args=(); env_args=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; env_args=("$@"); break; fi
  runner_args+=("$1"); shift
done
exec .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
  --enable_cameras \
  --kit_args "--/renderer/multiGpu/enabled=false --/crashreporter/enabled=false" \
  --external_environment_class_path arena_nurec.nurec_wormhole_env:NurecWormholeEnvironment \
  --policy_type zero_action --num_steps 60 --record_camera_video \
  "${runner_args[@]}" nurec_wormhole "${env_args[@]}"
