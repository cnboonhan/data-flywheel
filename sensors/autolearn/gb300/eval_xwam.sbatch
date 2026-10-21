#!/bin/bash
# Evaluate an X-WAM checkpoint on BEHAVIOR-1K picking_up_cup instances (the grasp at the counter) on one GB300 GPU:
# adapter/xwam/serve_xwam.py and adapter/eval_galaxea_behavior.py --space ee side by side, one simulator process per
# instance, as gb300/eval_galaxea.sbatch does for GalaxeaVLA.
#   sbatch sensors/autolearn/gb300/eval_xwam.sbatch <label> <experiment dir> <steps> <start dir> <seed> [<seed> ...]
# <experiment dir> is train_xwam.sbatch's $STATE/xwam/<run> (config.yaml + checkpoints/<steps>.ckpt/); <start dir> holds
# the scripted policy's runs on these instances (the base pose it parks at). Results: $STATE/eval/xwam/<label>/results.jsonl
# and a video per instance; the log in $STATE/logs/xwam-eval-<job>.log.
#SBATCH -p raus -N1 -n1 -c16 --gres=gpu:1 --mem=200G -t 08:00:00 -J xwam-eval
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/xwam-eval-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
set -a; . "$A/gb300/behavior.env"; set +a
label=$1; exp=$2; steps=$3; starts=$4; shift 4
out=$STATE/eval/xwam/$label
mkdir -p "$out"
gpu=${CUDA_VISIBLE_DEVICES%%,*}
port=$((18400 + gpu))
# the policy server on this job's GPU (behavior.env's preloads and JAX setting are for the simulator)
CUDA_VISIBLE_DEVICES=$gpu LD_PRELOAD= CUDA_HOME=$STATE/cuda-12.8 "$STATE/venv-xwam/bin/python" "$A/adapter/xwam/serve_xwam.py" \
  --exp "$exp" --steps "$steps" --port "$port" > "$out/server-$SLURM_JOB_ID.log" 2>&1 &
srv=$!; trap 'kill $srv 2>/dev/null' EXIT
for _ in $(seq 180); do curl -sf "http://127.0.0.1:$port/" >/dev/null && break; sleep 5; done   # T5, VAE and a 39 GB checkpoint: minutes
curl -sf "http://127.0.0.1:$port/" >/dev/null || { echo "policy server did not start: $out/server-$SLURM_JOB_ID.log"; exit 1; }
# the simulator as collect_behavior.sbatch runs it: every GPU visible, OmniGibson on this one, caches per GPU
export OMNIGIBSON_GPU_ID=$gpu CUDA_VISIBLE_DEVICES=0,1,2,3 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMNIGIBSON_APPDATA_PATH=$OMNIGIBSON_APPDATA_PATH-gpu$gpu HOME=/tmp/$USER/home-gpu$gpu
mkdir -p "$OMNIGIBSON_APPDATA_PATH" "$HOME"
echo "node $(hostname) GPU $gpu label $label exp $exp steps $steps seeds $*"
for seed in "$@"; do
  res=$(find "$starts"/seeds_*/seed_$(printf '%04d' "$seed") -name program-result.json 2>/dev/null | head -1)
  [[ -n $res ]] || { echo "seed $seed: no scripted run to start from"; continue; }
  grep -q "\"seed\": $seed," "$out/results.jsonl" 2>/dev/null && continue   # a rerun resumes
  cd "$A/aspire"
  timeout -k 30 1800 unshare --user --map-root-user --ipc --mount -- bash -c 'mount -t tmpfs -o size=32g tmpfs /dev/shm && exec "$0" "$@"' \
    "$STATE/venv-behavior/bin/python" "$A/adapter/eval_galaxea_behavior.py" --space ee --seed "$seed" --start "$res" \
    --server "http://127.0.0.1:$port" --out "$out/results.jsonl" --label "$label" --video "$out/seed_$(printf '%04d' "$seed").mp4" \
    > "$out/seed_$(printf '%04d' "$seed").log" 2>&1 || echo "seed $seed: runner exited $?"
  tail -1 "$out/results.jsonl" 2>/dev/null
done
