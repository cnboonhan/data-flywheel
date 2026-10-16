"""Does GPU memory sharing over the NVLink fabric work between two trays? One process per GPU (torchrun, 2 nodes x 4).

Each GPU allocates torch symmetric memory and exchanges handles with the other 7 (symm_mem.rendezvous: the call Kimi
K3's MoE tail fusion makes at TP8, which hangs on rack 02), writes its rank into its buffer, then reads every peer's
buffer directly and times a copy from a GPU on the other tray. Those reads are loads over NVLink, not network
transfers: they only work if the fabric maps the other tray's memory. Prints one verdict line from rank 0:

    PASS  fabric memory works across trays: read 8/8 peers, <n> GB/s from the other tray
    FAIL  <the error>
(and run.sbatch / run_docker.sh print HANG when the rendezvous never completes, which is how it fails on rack 02)
"""
import datetime
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

N = 64 << 20   # floats per buffer (256 MiB): large enough for a bandwidth figure, small next to a serving engine
rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
imex = os.path.exists("/dev/nvidia-caps-imex-channels/channel0")
if local == 0:
    print(f"{socket.gethostname()}: IMEX channel device {'visible' if imex else 'MISSING'}, torch {torch.__version__}", flush=True)
# NCCL is only the process group symm_mem needs; keep it off the fabric so only symmetric memory is tested
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90))
world = dist.get_world_size()
try:
    buf = symm_mem.empty(N, device="cuda")
    buf.fill_(rank)
    handle = symm_mem.rendezvous(buf, dist.group.WORLD)
    handle.barrier()
    seen = 0
    for peer in range(world):
        seen += int(handle.get_buffer(peer, (N,), torch.float32)[0].item() == peer)
    other = (rank + world // 2) % world   # a GPU on the other tray
    remote, local_copy = handle.get_buffer(other, (N,), torch.float32), torch.empty(N, device="cuda")
    local_copy.copy_(remote); torch.cuda.synchronize(); t = time.time()
    for _ in range(10):
        local_copy.copy_(remote)
    torch.cuda.synchronize()
    gbps = 10 * N * 4 / (time.time() - t) / 1e9
    result = torch.tensor([seen, gbps], device="cuda")
except Exception as e:   # noqa: BLE001: the verdict is the point
    print(f"FAIL  rank {rank}: {type(e).__name__}: {str(e)[:300]}", flush=True)
    os._exit(1)
gathered = [torch.zeros_like(result) for _ in range(world)]
dist.all_gather(gathered, result)
if rank == 0:
    worst = min(int(g[0]) for g in gathered)
    slowest = min(float(g[1]) for g in gathered)
    ok = worst == world
    print(f"{'PASS' if ok else 'FAIL'}  fabric memory {'works' if ok else 'is wrong'} across trays: every GPU read "
          f"{worst}/{world} peers correctly, >= {slowest:.0f} GB/s from the other tray", flush=True)
dist.destroy_process_group()
