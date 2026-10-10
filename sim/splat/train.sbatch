#!/bin/bash
#SBATCH --job-name=splat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --exclude=C2-GB300-02-C03
#SBATCH --output=%x-%j.log
# Train a splat with sim/splat/train.py on one GPU; arguments are train.py's. Excludes C03 (its GPUs serve Triton).
#   set -a; . $STATE_DIR/slurm.env; set +a; export SBATCH_PARTITION=<partition>
#   cd $STATE_DIR/slurm-logs && sbatch --export=ALL $FLYWHEEL_ROOT/sim/splat/train.sbatch <input> <output> [train.py args]
set -euo pipefail
: "${FLYWHEEL_ROOT:?source slurm.env and submit with --export=ALL}"
export PATH=$HOME/.local/bin:$PATH
cd "$FLYWHEEL_ROOT"
echo "splat on $(hostname): $*"
time uv run sim/splat/train.py "$@"
