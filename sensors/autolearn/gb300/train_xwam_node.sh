#!/bin/bash
# One node's share of train_xwam_multinode.sbatch (srun runs it once per node): a private /dev/shm and torchrun for this
# node's GPUs, joining the rendezvous on the first node. Everything else comes from the sbatch's environment.
set -euo pipefail
ngpu=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
mkdir -p "$XWAM_NODE_CACHE"
cd "$X"
# a /dev/shm of its own: logind removes the user's /dev/shm files whenever their last session on the node ends
exec unshare --user --map-root-user --ipc --mount -- bash -c 'mount -t tmpfs -o size=256g tmpfs /dev/shm && exec "$0" "$@"' \
  torchrun --nnodes="$SLURM_NNODES" --node_rank="$SLURM_NODEID" --nproc_per_node="$ngpu" \
  --master_addr="$XWAM_MASTER" --master_port="$XWAM_PORT" "$A/adapter/xwam/train_xwam.py" "$@"
