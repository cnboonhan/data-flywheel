#!/bin/bash
# BEHAVIOR-1K demonstrations on one GB300 GPU: adapter/collect_behavior.sh over a range of task instances, in the
# environment setup.sh builds on arm64. Several jobs over disjoint seed ranges collect in parallel.
#   sbatch sensors/autolearn/gb300/collect_behavior.sbatch <policy.py> <first seed> <last seed> [collect_behavior.sh options]
#   e.g. sbatch ... adapter/programs/pick_radio_r1pro.py 46 75 --target 30
# Runs land in $STATE/collect/behavior/<suite>/<policy>/seeds_<first>-<last>/ (collect.jsonl inside); the log in
# $STATE/logs/behavior-<job>.log.
#
# Runs the policy without the bubblewrap sandbox (the nodes have no bwrap): only for reviewed programs, never a
# campaign's agent-written ones.
#SBATCH -p raus -N1 -n1 -c16 --gres=gpu:1 --mem=128G -t 12:00:00 -J behavior-collect
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/behavior-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
set -a; . "$A/gb300/behavior.env"; set +a
policy=$(realpath "$1"); first=$2; last=$3; shift 3
suite=turning_on_radio; args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do [[ ${args[i]} == --suite ]] && suite=${args[i+1]}; done
# Slurm here sets CUDA_VISIBLE_DEVICES to the allocated GPU but leaves every GPU visible; Isaac's renderer counts all
# of them, so keep them all visible and point OmniGibson's render and physics GPU at the allocated one
export OMNIGIBSON_GPU_ID=${CUDA_VISIBLE_DEVICES%%,*} CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID
export BEHAVIOR_HARNESS=$A/gb300/run_behavior.sh
echo "node $(hostname) GPU $OMNIGIBSON_GPU_ID policy $policy seeds $first-$last"
"$A/adapter/collect_behavior.sh" "$policy" "$first" "$last" \
  --out "$STATE/collect/behavior/$suite/$(basename "${policy%.py}")/seeds_$first-$last" "${args[@]}"
