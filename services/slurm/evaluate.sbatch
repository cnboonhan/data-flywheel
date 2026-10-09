#!/usr/bin/env bash
#SBATCH --job-name=evaluate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log
# On C01 and C02 the eval client's Kit viewport stopped rendering a frame during the afternoon of
# 2026-10-08 (first with the ACT policy server beside it, later with the zero-action demo policy too);
# Isaac Lab then can't place its viewport camera. Identical software, GPU mode and scratch on all four
# nodes; cause not found. C03 and C04 work.
#SBATCH --exclude=C2-GB300-02-C01,C2-GB300-02-C02
#
# Evaluate a checkpoint of <model> for <embodiment> in simulation and record the result on its MLflow training run.
#   sbatch --export=ALL evaluate.sbatch <model> <embodiment> <task> <ckpt> [action] [seed] [eval_num]
#   e.g. sbatch --export=ALL evaluate.sbatch act arx_x5 stack_bowls RoboDojo-stack_bowls-arx_x5-joint-0 joint 0 5
# <model>: models/<model>/eval.sh. <ckpt>: the checkpoint name the model's recipe understands (ACT/DP: XPolicyLab's
# <bench>-<task>-<env_cfg>-<action>-<seed>, the training run's tag xpolicylab_ckpt, or a RoboDojo hub name).
set -euo pipefail
MODEL=${1:?model}; EMBODIMENT=${2:?embodiment}; TASK=${3:?task}; CKPT=${4:?ckpt}; ACTION=${5:-joint}; SEED=${6:-0}; EVAL_NUM=${7:-}

source "${FLYWHEEL_ROOT:-$HOME/workspaces/data-flywheel}/services/slurm/lib/common.sh"
load_model "$MODEL" eval
load_embodiment "$EMBODIMENT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
evaluate
