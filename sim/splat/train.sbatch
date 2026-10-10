#!/bin/bash
#SBATCH --job-name=splat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.log
# Train a splat with sim/splat/train.py on one GPU; arguments are train.py's. Submit with --exclude=$SERVICE_NODE: Triton holds that node's GPUs.
#   set -a; . $STATE_DIR/slurm.env; set +a; export SBATCH_PARTITION=<partition>
#   cd $STATE_DIR/slurm-logs && sbatch --export=ALL --exclude=$SERVICE_NODE $FLYWHEEL_ROOT/sim/splat/train.sbatch <input> <output> [train.py args]
set -euo pipefail
: "${FLYWHEEL_ROOT:?source slurm.env and submit with --export=ALL}"
export PATH=$HOME/.local/bin:$PATH
cd "$FLYWHEEL_ROOT"
echo "splat on $(hostname): $*"
time uv run sim/splat/train.py "$@"
