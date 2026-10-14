# slurm-exporter

Publishes which GPU each running Slurm job holds, `slurm_job_gpu{instance, gpu, job_id, user, job_name, partition}`, for [Prometheus](../prometheus/README.md) to join with the GPU metrics. It runs `scontrol` from the host's Slurm client (`SLURM_DIR`, `SLURM_BIN`, `SLURM_CONF` in `.env`) as `SERVICE_UID`. If the cluster hides other users' jobs (`PrivateData=jobs`), that account sees only its own; an account allowed to see every job can keep a file up to date instead, and the exporter serves that.

## Publish every job

Run the writer as the account that sees all jobs, on any host with the Slurm client (it needs only Python 3 and `scontrol` on `PATH`, or `SLURM_BIN`). `$STATE_DIR` is the stack's, from `.env`.

1. Check that the account is in the group that owns `$STATE_DIR` (the folder below is writable by that group) and sees other users' jobs.
   ```bash
   id -nG; stat -c %G $STATE_DIR/slurm-exporter; squeue -h | wc -l
   ```
2. Keep the writer running with Slurm's cron, so it survives logouts (`scrontab -e`, then add this line with `$STATE_DIR` written out; it restarts every 30 minutes and rewrites the file every 30 s):
   ```
   */30 * * * * timeout 1790 python3 $STATE_DIR/slurm-exporter/exporter.py --write $STATE_DIR/slurm-exporter/slurm_job_gpu.prom
   ```
   For a test, run the same command by hand.
3. Check that Prometheus gets it: `slurm_job_gpu_source{source="file"}` is 1 and `slurm_job_gpu_file_age_seconds` stays under 120.

## Notes

- The exporter serves the file while it is under 120 s old (`FRESH_S`, four missed rewrites); after that it falls back to asking Slurm as `SERVICE_UID` (`slurm_job_gpu_source{source="scontrol"}`), so a stopped writer shows as fewer jobs, not stale ones.
- `ctl.sh up` copies `exporter.py` to `$STATE_DIR/slurm-exporter/` for the writer, which may not be able to read the checkout.
- The file holds every running job's ID, user and name, readable by `$STATE_DIR`'s group; on a cluster that keeps jobs private, that is the information it hides, so publish it only with the cluster admins' agreement.
