#!/bin/bash
# Fabric-memory test on two trays of one rack, in Slurm containers (pyxis, as Slurm users run vLLM), vLLM 0.31's image:
#   sbatch -p <partition> -w <tray A>,<tray B> services/dynamo/fabric_test/run.sbatch
# Prints the trays' fabric state, then PASS, FAIL or HANG (README.md next to this file).
#SBATCH --nodes=2 --ntasks-per-node=1 --gres=gpu:4 --time=00:15:00 --job-name=fabric-test --output=fabric-test-%j.log
here=/home/htx_boonhan/workspaces/data-flywheel/services/dynamo/fabric_test   # sbatch runs a copy of this file
IMAGE=${IMAGE:-docker://vllm/vllm-openai:v0.31.0}   # pulled by pyxis on each tray (a few minutes the first time)
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST")); head=$(getent hosts "${nodes[0]}" | awk '{print $1}')
for n in "${nodes[@]}"; do
  echo "$n: $(srun -w "$n" -N1 -n1 bash -c 'nvidia-smi -q -i 0 | grep -A4 "^    Fabric" | grep -E "State|Status|ClusterUUID" | sed "s/  */ /g" | tr "\n" ";"')"
done
# NCCL only sets up the process group, over the network: the test is the symmetric memory, not NCCL's own fabric use
srun -N2 -n2 --container-image="$IMAGE" --container-mounts="$here:/test:ro" \
  --export=ALL,NCCL_MNNVL_ENABLE=0,NCCL_NVLS_ENABLE=0,NCCL_SOCKET_IFNAME=bond0,GLOO_SOCKET_IFNAME=bond0 \
  bash -c "timeout 240 torchrun --nnodes 2 --nproc-per-node 4 --node-rank \$SLURM_NODEID --master-addr $head \
           --master-port 29597 /test/probe.py 2>&1 | grep -E '^(PASS|FAIL)|IMEX channel'; exit \${PIPESTATUS[0]}" | tee out.$$
started=$(grep -c "IMEX channel" out.$$); verdict=$(grep -cE "^(PASS|FAIL)" out.$$); rm -f out.$$
if (( started < 2 )); then echo "ERROR the test didn't start on both trays (see the errors above); not a verdict, run it again"
elif (( verdict == 0 )); then echo "HANG  the GPUs never finished exchanging fabric memory handles (killed after 240 s): fabric memory doesn't work across these trays"
fi
exit 0
