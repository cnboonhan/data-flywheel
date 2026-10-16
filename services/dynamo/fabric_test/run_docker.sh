#!/bin/bash
# The same test in Docker, as the Dynamo engines run (dynamo/render.py), on two trays over ssh:
#   services/dynamo/fabric_test/run_docker.sh <tray A> <tray B>            the engines' devices (GPUs, InfiniBand)
#   IMEX=1 services/dynamo/fabric_test/run_docker.sh <tray A> <tray B>     plus the IMEX channel device
# Needs ~1 GiB free on each GPU; runs beside a serving engine. Prints PASS, FAIL or HANG (README.md next to this file).
set -u
here=$(cd "$(dirname "$0")" && pwd)
A=$1 B=$2; head=$(getent hosts "$A" | awk '{print $1}')
extra=(); [[ ${IMEX:-0} == 1 ]] && extra=(--device /dev/nvidia-caps-imex-channels/channel0)
run() {   # node rank
  ssh "$1" "bash -lc $(printf '%q' "docker run --rm --name fabric-test-$2 --network host --ipc host --shm-size 8g \
    --device nvidia.com/gpu=all --device /dev/infiniband ${extra[*]} --cap-add IPC_LOCK --ulimit memlock=-1 \
    -v $here:/test:ro -e NCCL_MNNVL_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 \
    --entrypoint timeout vllm/vllm-openai:v0.31.0 240 torchrun --nnodes 2 --nproc-per-node 4 --node-rank $2 \
    --master-addr $head --master-port 29598 /test/probe.py")" 2>&1 | grep -E '^(PASS|FAIL)|IMEX channel'
  return "${PIPESTATUS[0]}"
}
run "$B" 1 & run "$A" 0; rc=$?; wait
[[ $rc == 124 || $rc == 137 ]] && echo "HANG  the GPUs never finished exchanging fabric memory handles (killed after 240 s): fabric memory doesn't work here"
for n in "$A" "$B"; do ssh "$n" "bash -lc 'docker rm -f fabric-test-0 fabric-test-1'" >/dev/null 2>&1; done
exit 0
