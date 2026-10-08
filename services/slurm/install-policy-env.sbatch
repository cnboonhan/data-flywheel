#!/usr/bin/env bash
#SBATCH --job-name=install-policy-env
#SBATCH --partition=raus_manual
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
#
# Build (or re-point) the uv env an XPolicyLab policy trains and serves in: ENVS_DIR/<policy, lowercase>.
#   sbatch install-policy-env.sbatch ACT | DP | demo_policy
# Follows policy/<P>/install.sh, with the versions that work on this aarch64 Blackwell node: torch from
# the cu128 index (the policies pin torch==2.4.1, which has no CUDA build here), numpy/numba pins relaxed.
# Editable installs point at PROJECT_ROOT, so re-run this after moving the repo. Idempotent.
set -euo pipefail
policy=${1:?ACT|DP|demo_policy}
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/workspaces/data-flywheel/eval/system1/RoboDojo}
ENVS_DIR=${ENVS_DIR:-/tier1/htx_boonhan/services/envs}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tier1/htx_boonhan/services/uv-cache}
export PATH="$HOME/.local/bin:$PATH"
XPL=$PROJECT_ROOT/XPolicyLab
env=$ENVS_DIR/${policy,,}
torch=(torch torchvision --index-url https://download.pytorch.org/whl/cu128)

mkdir -p "$ENVS_DIR"
[[ -d $env ]] || uv venv --python 3.10 "$env"
source "$env/bin/activate"
echo "[$(date)] $policy env at $env, XPolicyLab at $XPL"
case $policy in
  ACT)
    uv pip install -q "${torch[@]}"
    uv pip install -q pyquaternion pyyaml rospkg pexpect opencv-python-headless matplotlib einops packaging h5py ipython numpy
    uv pip install -q -e "$XPL/policy/ACT/detr" --reinstall-package detr ;;
  DP)
    uv pip install -q "${torch[@]}"
    uv pip install -q zarr==2.12.0 wandb ipdb gpustat omegaconf hydra-core==1.2.0 dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 numba \
      moviepy imageio av matplotlib termcolor sympy h5py opencv-python-headless "numpy<2" huggingface_hub==0.25.2 pandas
    uv pip install -q -e "$XPL/policy/DP" --reinstall-package diffusion-policy ;;
  demo_policy) ;;
  *) echo "no recipe for $policy" >&2; exit 2 ;;
esac
uv pip install -q -e "$XPL" mlflow-skinny --reinstall-package xpolicylab
python -c "import XPolicyLab, sys; print('XPolicyLab from', XPolicyLab.__file__); sys.exit(0 if XPolicyLab.__file__.startswith('$XPL/') else 1)"
[[ $policy == demo_policy ]] || python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "$policy env ready"
