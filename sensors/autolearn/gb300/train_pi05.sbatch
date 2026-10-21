#!/bin/bash
# Fine-tune π0.5 (XPolicyLab's policy/Pi_05: openpi, JAX) from the pi05_base checkpoint on a BEHAVIOR-1K collection,
# on one GB300 node, data-parallel over the GPUs Slurm allocates.
#   sbatch sensors/autolearn/gb300/train_pi05.sbatch <run name> [openpi train flags...]
#   e.g. a save test: sbatch ... savetest --num-train-steps 20 --save-interval 10 --log-interval 5 --overwrite
# Needs gb300/setup_pi05.sh, the dataset from adapter/pi05/to_lerobot_v30.py and its stats from adapter/pi05/norm_stats.py.
# The config is adapter/pi05/r1pro.py's, registered by adapter/pi05/run.py; checkpoints go to $STATE/pi05/checkpoints/
# pi05_r1pro_picking_up_cup/<run name>/<step>/, not into the submodule.
#SBATCH -p raus -N1 -n1 -c128 --gres=gpu:4 --mem=800G -t 24:00:00 -J pi05-train
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/pi05-train-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
STATE=/tier1/mistral/flywheel/autolearn
run=$1; shift
export STATE HF_LEROBOT_HOME=$STATE/pi05/lerobot HF_HUB_OFFLINE=1 OPENPI_DATA_HOME=$STATE/models/openpi WANDB_MODE=disabled
# jaxlib 0.5.3 does not know compute capability 10.3 (GB300): it targets sm_101, where its Triton GEMMs emit tcgen05
# instructions ptxas rejects; with Triton GEMMs off the matmuls go to cuBLAS, which supports the GPU
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9    # XPolicyLab's Pi_05 train.sh
export PYTHONUNBUFFERED=1    # the loss lines go to stdout, which a batch job would otherwise buffer until exit
echo "node $(hostname) GPUs $CUDA_VISIBLE_DEVICES run $run"
cd "$A/adapter/pi05"
# a /dev/shm of its own: logind removes the user's /dev/shm files whenever their last session on the node ends, and the
# loader workers pass batches through it. 96 workers: a sample decodes three AV1 frames in ~0.2 s, so 128 per step take
# ~25 CPU-seconds
unshare --user --map-root-user --ipc --mount -- bash -c 'mount -t tmpfs -o size=256g tmpfs /dev/shm && exec "$0" "$@"' \
  "$STATE/venv-pi05/bin/python" run.py train pi05_r1pro_picking_up_cup --exp-name "$run" --num-workers 96 "$@"
