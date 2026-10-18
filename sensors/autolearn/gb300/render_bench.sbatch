#!/bin/bash
# Genie Sim's rendering on a GB300 GPU: render_bench.py in NVIDIA's Isaac Sim 5.1.0 image (arm64), on the Slurm-allocated GPU.
#   sbatch -w <node> sensors/autolearn/gb300/render_bench.sbatch
# Results: $STATE/logs/render-<job>.log and $STATE/results/render-<job>-*.json; Isaac's caches (shaders compile on the
# first run, minutes) persist in $STATE/isaac-cache, so later runs start faster.
#SBATCH -p raus -N1 -n1 -c16 --gres=gpu:1 --mem=200G -t 03:00:00 -J geniesim-render
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/render-%j.log
set -euo pipefail
STATE=/tier1/mistral/flywheel/autolearn
HERE=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn/gb300   # sbatch runs a copy of this file
IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
GPU=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}; GPU=${GPU%%,*}   # the allocated GPU's index, as the node's CDI names it
mkdir -p "$STATE"/results "$STATE"/isaac-cache/{kit,ov,pip,glcache,computecache,logs,data,documents}
echo "node $(hostname) GPU $GPU image $IMAGE"; nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader -i "$GPU"
docker pull -q "$IMAGE"
run() {   # renderer cameras width height
  docker run --rm --name "geniesim-render-$SLURM_JOB_ID" --network host --device "nvidia.com/gpu=$GPU" \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
    -v "$STATE/isaac-cache/kit:/isaac-sim/kit/cache:rw" -v "$STATE/isaac-cache/ov:/root/.cache/ov:rw" \
    -v "$STATE/isaac-cache/pip:/root/.cache/pip:rw" -v "$STATE/isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    -v "$STATE/isaac-cache/computecache:/root/.nv/ComputeCache:rw" -v "$STATE/isaac-cache/logs:/root/.nvidia-omniverse/logs:rw" \
    -v "$STATE/isaac-cache/data:/root/.local/share/ov/data:rw" -v "$STATE/isaac-cache/documents:/root/Documents:rw" \
    -v "$HERE:/bench:ro" -v "$STATE/results:/results:rw" \
    "$IMAGE" timeout 50m /isaac-sim/python.sh /bench/render_bench.py --renderer "$1" --cameras "$2" --width "$3" --height "$4" \
      --out "/results/render-$SLURM_JOB_ID-$1-$2x$3x$4.json" 2>&1 | grep -vE "^\[[0-9.]+s\] \[(Info|Warning)\]" || echo "RUN FAILED: $*"
}
run RealTimePathTracing 1 640 480      # the lightest case Genie Sim would use
run RealTimePathTracing 3 1280 720     # head + two wrist cameras at HD, as data collection
run RaytracedLighting 3 1280 720       # the cheaper renderer, for comparison
echo "RESULTS:"; grep -h "^RESULT" "$STATE/logs/render-$SLURM_JOB_ID.log" || true
