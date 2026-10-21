#!/bin/bash
# Evaluate a FastWAM checkpoint on BEHAVIOR-1K picking_up_cup instances (the grasp at the counter) on one GB300 GPU:
# adapter/fastwam/serve_fastwam.py and adapter/eval_galaxea_behavior.py --space joint side by side, one simulator process
# per instance, as gb300/eval_pi05.sbatch.
#   sbatch sensors/autolearn/gb300/eval_fastwam.sbatch <label> <ckpt> <start dir> <seed> [<seed> ...]
# <ckpt> is a training checkpoint ($STATE/fastwam/<run>/run/checkpoints/weights/step_NNNNNN.pt); the normalization stats
# are the ones it trained with ($FASTWAM_STATS, default the floored cup stats). <start dir> holds the scripted policy's
# runs on these instances. Results: $STATE/eval/fastwam/<label>/results.jsonl and a video per instance; the log in
# $STATE/logs/fastwam-eval-<job>.log.
#SBATCH -p raus -N1 -n1 -c16 --gres=gpu:1 --mem=200G -t 08:00:00 -J fastwam-eval
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/fastwam-eval-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
set -a; . "$A/gb300/behavior.env"; set +a
label=$1; ckpt=$2; starts=$3; shift 3
out=$STATE/eval/fastwam/$label
stats=${FASTWAM_STATS:-$STATE/fastwam/dataset_stats_fastwam_picking_up_cup.json}
mkdir -p "$out"
gpu=${CUDA_VISIBLE_DEVICES%%,*}
port=$((18300 + gpu))
# the policy server on this job's GPU (bf16 inference, ~12 GB; the simulator shares the GPU); Wan2.2's text encoder and
# VAE from the local model tree, never downloaded
CUDA_VISIBLE_DEVICES=$gpu LD_PRELOAD= DIFFSYNTH_MODEL_BASE_PATH=$STATE/models/diffsynth DIFFSYNTH_SKIP_DOWNLOAD=true \
  CUDA_HOME=$STATE/cuda-12.8 HF_HUB_OFFLINE=1 \
  "$STATE/venv-fastwam/bin/python" "$A/adapter/fastwam/serve_fastwam.py" --ckpt "$ckpt" --stats "$stats" \
  --action-dit "$STATE/fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt" --port "$port" > "$out/server-$SLURM_JOB_ID.log" 2>&1 &
srv=$!; trap 'kill $srv 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$port/" >/dev/null && break; sleep 5; done   # ~2-3 min to load
curl -sf "http://127.0.0.1:$port/" >/dev/null || { echo "policy server did not start: $out/server-$SLURM_JOB_ID.log"; exit 1; }
# the simulator as collect_behavior.sbatch runs it: every GPU visible, OmniGibson on this one, caches per GPU
export OMNIGIBSON_GPU_ID=$gpu CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMNIGIBSON_APPDATA_PATH=$OMNIGIBSON_APPDATA_PATH-gpu$gpu HOME=/tmp/$USER/home-gpu$gpu
mkdir -p "$OMNIGIBSON_APPDATA_PATH" "$HOME"
echo "node $(hostname) GPU $gpu label $label ckpt $ckpt seeds $*"
for seed in "$@"; do
  res=$(find "$starts"/seeds_*/seed_$(printf '%04d' "$seed") -name program-result.json 2>/dev/null | head -1)
  [[ -n $res ]] || { echo "seed $seed: no scripted run to start from"; continue; }
  grep -q "\"seed\": $seed," "$out/results.jsonl" 2>/dev/null && continue   # a rerun resumes
  cd "$A/aspire"
  timeout -k 30 1800 unshare --user --map-root-user --ipc --mount -- bash -c 'mount -t tmpfs -o size=32g tmpfs /dev/shm && exec "$0" "$@"' \
    "$STATE/venv-behavior/bin/python" "$A/adapter/eval_galaxea_behavior.py" --space joint --seed "$seed" --start "$res" \
    --server "http://127.0.0.1:$port" --out "$out/results.jsonl" --label "$label" --video "$out/seed_$(printf '%04d' "$seed").mp4" \
    > "$out/seed_$(printf '%04d' "$seed").log" 2>&1 || echo "seed $seed: runner exited $?"
  tail -1 "$out/results.jsonl" 2>/dev/null
done
