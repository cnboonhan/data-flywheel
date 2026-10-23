#!/bin/bash
# BEHAVIOR-1K demonstrations on one GB300 GPU: a SAM3 service and adapter/collect_behavior.sh over a range of task
# instances, both on the job's GPU, in the environments setup.sh builds on arm64. Several jobs over disjoint seed
# ranges collect in parallel (gb300/collect_parallel.sh submits them).
#   sbatch sensors/autolearn/gb300/collect_behavior.sbatch <policy.py> <first seed> <last seed> [collect_behavior.sh options]
#   e.g. sbatch ... adapter/programs/pick_cup_r1pro.py 68 97 --target 30 --suite picking_up_cup
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
suite=picking_up_cup; args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do [[ ${args[i]} == --suite ]] && suite=${args[i+1]}; done
# Slurm here sets CUDA_VISIBLE_DEVICES to the allocated GPU but leaves every GPU visible; Isaac's renderer counts all
# of them, so keep them all visible and point OmniGibson's render and physics GPU at the allocated one
export OMNIGIBSON_GPU_ID=${CUDA_VISIBLE_DEVICES%%,*} CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID
# Isaac's caches and lock files node-local and one set per GPU: in the NFS home (~/.cache/ov, ~/.nv, ~/.cache/warp) or
# shared between concurrent runs, start-up hangs after "app ready" and kitchen-scene materials fail to load. Per GPU,
# the next job on that GPU still finds its shaders compiled
export OMNIGIBSON_APPDATA_PATH=$OMNIGIBSON_APPDATA_PATH-gpu$OMNIGIBSON_GPU_ID HOME=/tmp/$USER/home-gpu$OMNIGIBSON_GPU_ID
mkdir -p "$OMNIGIBSON_APPDATA_PATH" "$HOME"
export BEHAVIOR_HARNESS=$A/gb300/run_behavior.sh
# This job's own SAM3 (the policies' segment_text and localize_object), on its GPU; port 18114 + GPU index, so the up to
# four jobs on a node don't collide
port=$((18114 + OMNIGIBSON_GPU_ID))
HF_HOME=$STATE/hf-cache CUDA_VISIBLE_DEVICES=$OMNIGIBSON_GPU_ID LD_PRELOAD= \
  "$STATE/venv-sam3/bin/python" -m cap_harness.providers.sam3.service --port "$port" > "$STATE/logs/sam3-$SLURM_JOB_ID.log" 2>&1 &
sam3=$!; trap 'kill $sam3 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$port/healthz" >/dev/null && break; sleep 5; done   # loads in ~1 min
curl -sf "http://127.0.0.1:$port/healthz" >/dev/null || { echo "SAM3 did not start; see $STATE/logs/sam3-$SLURM_JOB_ID.log"; exit 1; }
export CAP_HARNESS_SAM3_URL=http://127.0.0.1:$port
echo "node $(hostname) GPU $OMNIGIBSON_GPU_ID policy $policy seeds $first-$last"
"$A/adapter/collect_behavior.sh" "$policy" "$first" "$last" \
  --out "$STATE/collect/behavior/$suite/$(basename "${policy%.py}")/seeds_$first-$last" "${args[@]}"
