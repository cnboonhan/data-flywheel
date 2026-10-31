#!/usr/bin/env bash
#SBATCH --job-name=evaluate
# Resources: 1 GPU shared by the simulator and the policy server (--gres=gpu:2 gives the server its own). CPUs and
# memory are untuned. 12 h is generous (2 episodes of 800 steps take ~10 min) and untuned.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
# Nodes whose Kit viewport doesn't render (Isaac Lab then can't place its viewport camera; cause not found) are kept
# out by submitting with ${EVAL_EXCLUDE:+--exclude=$EVAL_EXCLUDE} (EVAL_EXCLUDE in services/.env, via flywheel.env).
#
# Evaluate a checkpoint of <model> for <embodiment> in simulation and record the result on its MLflow training run.
#   sbatch --export=ALL evaluate.sbatch <model> <embodiment> <task> <ckpt> [action] [seed] [eval_num]
#   e.g. sbatch --export=ALL ${EVAL_EXCLUDE:+--exclude=$EVAL_EXCLUDE} evaluate.sbatch act arx_x5 stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 joint 0 5
# <model>: models/<model>/eval.sh. <ckpt>: the checkpoint name the model's recipe understands (ACT/DP: XPolicyLab's
# <bench>-<task>-<env_cfg>-<action>-<seed>, the training run's tag xpolicylab_ckpt, or a RoboDojo hub name).
set -euo pipefail
# EVAL_NUM empty: RoboDojo's own episode count for the task.
MODEL=${1:?model}; EMBODIMENT=${2:?embodiment}; TASK=${3:?task}; CKPT=${4:?ckpt}; ACTION=${5:-joint}; SEED=${6:-0}; EVAL_NUM=${7:-}

source "${FLYWHEEL_ROOT:-$HOME/workspaces/data-flywheel}/services/slurm/lib/common.sh"
load_model "$MODEL" eval
load_embodiment "$EMBODIMENT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}   # 0 only outside a Slurm GPU allocation
evaluate
