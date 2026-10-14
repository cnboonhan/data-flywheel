# slurm-exporter

`exporter.py` pushes which GPU each running Slurm job holds, `slurm_job_gpu{instance, gpu, job_id, user, job_name, partition}`, to the stack's Pushgateway, so [Prometheus](../prometheus/README.md) can join it with the GPU metrics. It sees the jobs of whichever account runs it: if the cluster hides other users' jobs (`PrivateData=jobs`), run it as an account allowed to see every job to publish them all.

## Push the jobs

It needs only Python 3 and `scontrol` on `PATH` (or `SLURM_BIN`), so run it on any host with the Slurm client. `ctl.sh up` copies it to `$STATE_DIR/slurm-exporter/exporter.py`, readable by `$STATE_DIR`'s group; any copy works.

1. Get `PUSHGATEWAY_TOKEN` from the admin (`services/.env`) and the stack's CA certificate ([quickstart](../../quickstart.md) step 3).
2. Check it pushes (stop it with Ctrl-C; that also deletes what it pushed).
   ```bash
   export PUSHGATEWAY_TOKEN=<token> PUSHGATEWAY_CA=<path to the CA certificate>
   python3 exporter.py --push https://$SERVICE_HOST:$CADDY_PORT/pushgateway
   ```
3. Keep it running with Slurm's cron, so it survives logouts (`scrontab -e`; it restarts every 30 minutes and pushes every 30 s). Put the token in a file only you can read rather than in the crontab:
   ```
   */30 * * * * . ~/.pushgateway.env && timeout 1790 python3 <path>/exporter.py --push https://<SERVICE_HOST>:<CADDY_PORT>/pushgateway
   ```
4. Check that Prometheus gets it: `time() - push_time_seconds{job="slurm_jobs"}` stays under 60 and `slurm_job_gpu` lists the running jobs.

## Notes

- Each push replaces the whole group `job="slurm_jobs"`, and a clean stop (SIGTERM from `timeout`, Ctrl-C) deletes it. A pusher that is killed leaves its last push in place; filter on `push_time_seconds` to ignore an old one.
- Without the token, or with another method or path, the push route doesn't match and the request gets a 404.
- What it pushes holds every visible job's ID, user and name, readable by everyone with Grafana access; on a cluster that keeps jobs private, push others' jobs only with the cluster admins' agreement.
