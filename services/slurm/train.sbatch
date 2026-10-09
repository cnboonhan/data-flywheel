#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.log
#
# Train <model> for <embodiment> on xspark hdf5 data, tracked in MLflow (experiment train/<embodiment>/<mix>,
# registered model <model>.<embodiment>.<mix>; lib/flywheel_mlflow.py).
#   sbatch --export=ALL train.sbatch <model> <embodiment> <data>[,<data>...] [--mix NAME] [--action joint|ee] [--seed N] [-- model args]
#   e.g. sbatch --export=ALL train.sbatch act arx_x5 RoboDojo/stack_bowls -- --num_epochs 30 --save_freq 30
#        sbatch --export=ALL train.sbatch mlp arx_x5 'RoboDojo/*,galaxeaOpenWorldDataset/*' --mix arx_x5-all -- --epochs 200
# <model>: models/<model>/train.sh. <embodiment>: embodiments/<embodiment>.yaml. <data>: <bench>/<task> globs under
# processed/xpolicylab; each dataset's episodes are in <bench>/<task>/<embodiment's data_env_cfg>/data/.
set -euo pipefail
MODEL=${1:?model, e.g. act}; EMBODIMENT=${2:?embodiment, e.g. arx_x5}; IFS=, read -ra DATA <<< "${3:?data, e.g. RoboDojo/stack_bowls}"
shift 3
MIX=""; ACTION=joint; SEED=0; EXTRA=()
while (($#)); do
  case $1 in
    --mix) MIX=$2; shift 2 ;;
    --action) ACTION=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown option $1 (model arguments go after --)" >&2; exit 2 ;;
  esac
done

source "${FLYWHEEL_ROOT:-$HOME/workspaces/data-flywheel}/services/slurm/lib/common.sh"
load_model "$MODEL" train
load_embodiment "$EMBODIMENT"
# shellcheck source=/dev/null
source "$ENVS_DIR/$ENV/bin/activate"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} WANDB_MODE=offline
echo "[$(date)] train $MODEL for $EMB_NAME on ${DATA[*]} (mix ${MIX:-default}, $ACTION, seed $SEED) on $(hostname), GPU $CUDA_VISIBLE_DEVICES, env $ENV"
train
