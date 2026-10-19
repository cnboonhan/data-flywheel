# nvlink_test

Checks NVLink between the GPUs of one tray: a 1 GiB copy from every GPU to every other, each byte checked, each pair in a process of its own. Run it before serving a model on a tray, and after an `uncorrectable NVLink error` or an NVLink Xid (144–150).

1. Run it on the tray, with its GPUs idle (a pass takes ~2 min).
   ```bash
   services/dynamo/nvlink_test/run.sh <tray>
   ```
2. Read one line per pair:

| Line | Meaning |
|---|---|
| `a->b ~230-290 GB/s OK` | the link works |
| `a->b FAIL CUDA error: uncorrectable NVLink error ...` | the link is down; an engine on these GPUs dies with the same error |
| `a->b ~16 GB/s WRONG DATA` | the copy completes with wrong bytes: an engine would serve garbage without failing |

Any line but `OK`: don't place an engine on that pair; ask the cluster admins to look at the tray, and run this again after their fix.

**Notes**
- The last Xid of each GPU is on the node's dcgm-exporter: `curl -s http://<tray>:9400/metrics | grep XID`. 145 is an NVLink error; 45 (a process was killed) and 31 (a page fault) come with any crash.
- `nvidia-smi nvlink -s` shows the links up even when this test fails them.
