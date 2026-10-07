#!/usr/bin/env bash
# Reconstruct a Gaussian splat from a COLMAP dataset with 3DGRUT (NuRec mono workflow) and export it as a
# ParticleField USD for Isaac Sim 6.x. Runs under sim/nurec/runs/<name>/.
#   bash sim/nurec/train.sh <colmap_dir> <name> [extra hydra overrides...]
#   bash sim/nurec/train.sh datasets/nurec-zh_lounge/zh_lounge/colmap zh_lounge
#   bash sim/nurec/train.sh ... zh_lounge_fast n_iterations=7000      # quick check
# <colmap_dir> must contain sparse/0/{cameras,images,points3D}.bin and images/. Every 8th image is held out
# and scored (PSNR/SSIM/LPIPS) after training. Needs the 3DGRUT venv: see sim/nurec/README.md.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRUT="${ROOT}/sim/nurec/3dgrut"
colmap_dir="$(realpath "${1:?usage: nurec_train.sh <colmap_dir> <name> [overrides...]}")"; shift
name="${1:?usage: nurec_train.sh <colmap_dir> <name> [overrides...]}"; shift
[ -d "${colmap_dir}/sparse/0" ] || { echo "no sparse/0 in ${colmap_dir}" >&2; exit 1; }
[ -x "${GRUT}/.venv/bin/python" ] || { echo "3DGRUT venv missing; see sim/nurec/README.md" >&2; exit 1; }

cd "${GRUT}"
source .venv/bin/activate   # also exports the venv's CUDA toolkit paths for the JIT-compiled kernels
exec python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
  path="${colmap_dir}" out_dir="${ROOT}/sim/nurec/runs" experiment_name="${name}" \
  export_usd.enabled=true "$@"
