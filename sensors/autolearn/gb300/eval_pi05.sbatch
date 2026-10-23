#!/bin/bash
# Evaluate a π0.5 checkpoint on BEHAVIOR-1K picking_up_cup instances (the grasp at the counter) on one GB300 GPU:
# adapter/pi05/serve_pi05.py and adapter/pi05/eval_behavior.py side by side, one simulator process per instance.
#   sbatch sensors/autolearn/gb300/eval_pi05.sbatch <label> <ckpt> <start dir> <seed> [<seed> ...]
# <ckpt> is a training step directory ($STATE/pi05/checkpoints/<config>/<run>/<step>) or, for the base model, the
# pi05_base directory under $STATE/models/openpi. <start dir> holds the scripted policy's runs on these instances.
# Results: $STATE/eval/pi05/<label>/results.jsonl and a video per instance; the log in $STATE/logs/pi05-eval-<job>.log.
#SBATCH -p raus -N1 -n1 -c16 --gres=gpu:1 --mem=160G -t 08:00:00 -J pi05-eval
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/pi05-eval-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
set -a; . "$A/gb300/behavior.env"; set +a
label=$1; ckpt=$2; starts=$3; shift 3
out=$STATE/eval/pi05/$label
mkdir -p "$out"
gpu=${CUDA_VISIBLE_DEVICES%%,*}
port=$((18400 + gpu))
# the policy server on this job's GPU; JAX allocates as it goes (no 75% preallocation), the simulator shares the GPU.
# Triton GEMMs off: jaxlib 0.5.3 does not know the GB300's compute capability 10.3 (see train_pi05.sbatch)
CUDA_VISIBLE_DEVICES=$gpu LD_PRELOAD= STATE=$STATE OPENPI_DATA_HOME=$STATE/models/openpi HF_LEROBOT_HOME=$STATE/pi05/lerobot \
  HF_HUB_OFFLINE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
  "$STATE/venv-pi05/bin/python" "$A/adapter/pi05/serve_pi05.py" --ckpt "$ckpt" --port "$port" > "$out/server-$SLURM_JOB_ID.log" 2>&1 &
srv=$!; trap 'kill $srv 2>/dev/null' EXIT
for _ in $(seq 120); do curl -sf "http://127.0.0.1:$port/" >/dev/null && break; sleep 5; done   # ~1-2 min to load
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
    "$STATE/venv-behavior/bin/python" "$A/adapter/pi05/eval_behavior.py" --seed "$seed" --start "$res" \
    --server "http://127.0.0.1:$port" --out "$out/results.jsonl" --label "$label" --video "$out/seed_$(printf '%04d' "$seed").mp4" \
    > "$out/seed_$(printf '%04d' "$seed").log" 2>&1 || echo "seed $seed: runner exited $?"
  tail -1 "$out/results.jsonl" 2>/dev/null
done
