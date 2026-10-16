# fabric_test

Checks whether GPUs on two trays of one NVL72 rack can share memory over the NVLink fabric: each of the 8 GPUs allocates torch symmetric memory, exchanges handles with the others (`symm_mem.rendezvous`, the call Kimi K3's MoE tail fusion makes at TP8), reads every peer's buffer and times reads from the other tray. Those reads are NVLink loads, so they only work when the fabric maps the other tray's memory.

1. Run it on two trays of one rack, in Slurm containers (as Slurm users run vLLM):
   ```bash
   sbatch -p <partition> -w <tray A>,<tray B> services/dynamo/fabric_test/run.sbatch
   ```
   Expect ~3 min (more when a tray first pulls the image); the result is in `fabric-test-<job>.log` in the submit directory.
2. Or in Docker, as the Dynamo engines run (`IMEX=1` adds the IMEX channel device they lack):
   ```bash
   services/dynamo/fabric_test/run_docker.sh <tray A> <tray B>
   ```

| Verdict | Meaning |
|---|---|
| `PASS ... >= N GB/s` | fabric memory works between the trays (rack 01 trays: ~230 GB/s) |
| `HANG` | the handle exchange never completes: the fabric doesn't map the other tray's memory |
| `FAIL ... Failed to send fd` | no IMEX channel device in the container: torch falls back to file-descriptor sharing, which only works within one tray |
| `ERROR` | a tray didn't start the test (image pull, ...): not a verdict, run it again |

## Notes

- Results so far: rack 01 trays PASS; rack 02 trays (02-C03 + 02-C04) HANG, with the same image, launch and a healthy fabric state on both racks (`nvidia-smi -q`: Fabric Completed/Success; `nvidia-imex-ctl -N`: all 18 trays connected). The rack 02 runs shared the GPUs with a serving engine.
- Each line before the verdict shows the tray's fabric state and whether the container sees `/dev/nvidia-caps-imex-channels/channel0`.
- It allocates 256 MiB per GPU, so it can run beside a serving model.
