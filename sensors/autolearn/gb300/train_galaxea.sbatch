#!/bin/bash
# Fine-tune GalaxeaVLA's G0Tiny on a BEHAVIOR-1K collection with XPolicyLab's GalaxeaVLA train.sh, on one GB300 node.
#   sbatch sensors/autolearn/gb300/train_galaxea.sbatch <galaxea dataset dir> <run name> [hydra overrides...]
#   e.g. sbatch ... /tier1/mistral/flywheel/autolearn/datasets/galaxea_picking_up_cup cup_grasp
# The dataset comes from adapter/behavior_to_lerobot.py --format galaxea: one LeRobot root, or a directory of part_*
# roots converted in parallel (all are listed in data.dataset.dataset_dirs); the task config is GALAXEA_TASK_CONFIG
# (default r1pro_sim/g0tiny_behavior_grasp, from adapter/galaxea/). Checkpoints, stats and caches go under
# $STATE/galaxea/<run name>/, not into the submodule. setup.sh galaxea builds the venv; GalaxeaVLA/.venv links to it.
#SBATCH -p raus -N1 -n1 -c64 --gres=gpu:4 --mem=512G -t 12:00:00 -J galaxea-train
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/galaxea-train-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
STATE=/tier1/mistral/flywheel/autolearn
data=$(realpath "$1"); run=$2; shift 2
out=$STATE/galaxea/$run
mkdir -p "$out"
export GALAXEA_FM_OUTPUT_DIR=$out/checkpoints GALAXEA_FM_DATASET_STATS_CACHE_DIR=$out/stats HF_DATASETS_CACHE=$out/hf-datasets
parts=("$data"/part_*); extra=()
if [[ -d ${parts[0]} ]]; then      # train.sh checks one root; the later dataset_dirs override lists every part
  extra=("data.dataset.dataset_dirs=[$(IFS=,; echo "${parts[*]}")]"); data=${parts[0]}
fi
export GALAXEA_DATASET_DIR=$data GALAXEA_PRETRAINED_CKPT=$STATE/models/G0Tiny_260120/G0Tiny_260120
export GALAXEA_PALIGEMMA_PATH=$STATE/models/SmolVLM2-500M-Video-Instruct
export GALAXEA_TASK_CONFIG=${GALAXEA_TASK_CONFIG:-r1pro_sim/g0tiny_behavior_grasp}
gpus=$CUDA_VISIBLE_DEVICES      # the GPUs Slurm allocated (it sets this, though every GPU stays visible)
cd "$A/../../eval/system1/RoboDojo/XPolicyLab/policy/GalaxeaVLA"
echo "node $(hostname) GPUs $gpus data $data run $out task $GALAXEA_TASK_CONFIG"
# in a /dev/shm of its own (an unprivileged user namespace, as run_behavior.sh): logind removes the user's /dev/shm
# files whenever their last login session on the node ends (the services' monitor ssh-es in every minute), which killed
# a DataLoader worker's shared tensor mid-epoch and hung the run
unshare --user --map-root-user --ipc --mount -- bash -c 'mount -t tmpfs -o size=256g tmpfs /dev/shm && exec "$0" "$@"' \
  bash train.sh behavior "$run" r1pro joint 0 "$gpus" "hydra.searchpath=[file://$A/adapter/galaxea]" "${extra[@]}" "$@"
