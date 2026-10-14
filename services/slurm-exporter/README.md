# slurm-exporter

`exporter.py` pushes what each running Slurm job holds and uses to the stack's Pushgateway, so [Prometheus](../prometheus/README.md) can join it with the nodes' metrics:

| Metric | What | Joins |
|---|---|---|
| `slurm_job_gpu{instance, gpu, <job>}` | each GPU the job holds | dcgm-exporter's GPU metrics, `on (instance, gpu)` |
| `slurm_job_cpu{instance, cpu, <job>}` | each CPU core the job is pinned to | node_exporter's `node_cpu_seconds_total`, `on (instance, cpu)` |
| `slurm_job_cpus{<job>}`, `slurm_job_memory_bytes{<job>}` | CPUs and memory allocated (0 bytes: none requested) | |
| `slurm_job_rss_bytes{<job>}` | memory in use, summed over its running steps (`sstat`) | |

`<job>` is `job_id, user, job_name, partition`. It sees the jobs of whichever account runs it: if the cluster hides other users' jobs (`PrivateData=jobs`), run it as an account allowed to see every job to publish them all.

## Push the jobs

It needs only Python 3, `scontrol` and `sstat` on `PATH` (or `SLURM_BIN`), so run it on any host with the Slurm client. `ctl.sh up` copies it to `$STATE_DIR/slurm-exporter/exporter.py`, readable by `$STATE_DIR`'s group; any copy works.

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

- `cpu` is Slurm's `CPU_IDs`, which are the OS's core numbers where the two numberings agree (they do when a job's `Cpus_allowed_list` in `/proc/<pid>/status` matches its `CPU_IDs`). `sstat` reports memory as Slurm last sampled it (`JobAcctGatherFrequency`). CPU use comes from node_exporter's per-core counters instead of `sstat`'s cumulative CPU time: a rate per core, live, and no extra call per job.
- Each push replaces the whole group `job="slurm_jobs"` (`PUSHGATEWAY_JOB` to push elsewhere, e.g. for a test), and a clean stop (SIGTERM from `timeout`, Ctrl-C) deletes it. A pusher that is killed leaves its last push in place; filter on `push_time_seconds` to ignore an old one.
- Without the token, or with another method or path, the push route doesn't match and the request gets a 404.
- What it pushes holds every visible job's ID, user and name, readable by everyone with Grafana access; on a cluster that keeps jobs private, push others' jobs only with the cluster admins' agreement.
