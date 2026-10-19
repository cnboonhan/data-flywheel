"""One NVLink peer copy between two GPUs of a tray: python3 pair.py <from> <to>.

Copies 1 GiB from GPU <from> to GPU <to>, checks every byte and times the second copy (the first warms up). Run one
process per pair: an NVLink error poisons the process's CUDA context, so later pairs in it would fail too.
"""
import sys
import time

import torch

N = 1 << 28   # 256 Mi float32 = 1 GiB: long enough to time a link, small next to a GB300's 276 GiB

a, b = int(sys.argv[1]), int(sys.argv[2])
try:
    src = torch.arange(N, device=f"cuda:{a}", dtype=torch.float32)
    dst = src.to(f"cuda:{b}")
    torch.cuda.synchronize(b)
    t = time.time()
    dst = src.to(f"cuda:{b}")
    torch.cuda.synchronize(b)
    gbs = 4 * N / (time.time() - t) / 1e9
    print(f"{a}->{b} {gbs:.0f} GB/s {'OK' if torch.equal(dst.cpu(), src.cpu()) else 'WRONG DATA'}")
except Exception as e:   # noqa: BLE001  report the pair, let the runner go on to the next
    print(f"{a}->{b} FAIL {str(e).splitlines()[0]}")
