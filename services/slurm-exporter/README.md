# slurm-exporter

`exporter.py` pushes what each running Slurm job holds and uses, and each node's state and latest job, to the stack's Pushgateway, so [Prometheus](../prometheus/README.md) can join it with the nodes' metrics:

| Metric | What | Joins |
|---|---|---|
| `slurm_job_gpu{instance, gpu, <job>}` | each GPU the job holds | dcgm-exporter's GPU metrics, `on (instance, gpu)` |
| `slurm_job_cpu{instance, cpu, <job>}` | each CPU core the job is pinned to | node_exporter's `node_cpu_seconds_total`, `on (instance, cpu)` |
| `slurm_job_cpus{<job>}`, `slurm_job_memory_bytes{<job>}` | CPUs and memory allocated (0 bytes: none requested) | |
| `slurm_job_rss_bytes{<job>}` | memory in use, summed over its running steps (`sstat`) | |
| `slurm_node_state{instance, state, reason, partition}` | each node's state (`sinfo`: idle, mixed, allocated, drained, down, ...; a trailing `*`: not responding) and why | node metrics, `on (instance)` |
| `slurm_node_last_job_start_seconds{instance, <job>, job_state}`, `..._end_seconds` | the latest job started on the node (`sacct`, last 30 days) and when it started and ended (0 while it runs) | |
| `slurm_node_reason_seconds{instance}` | when Slurm set a down or drained node's reason (`sinfo %H`) | |
| `slurm_node_cause_job_seconds{instance, cause_job, cause_user, cause_name, cause_state}` | the job it went down on: the one Slurm requeued or failed on it then (`sacct -D`: a Prolog error requeues the job it was starting, NODE_FAIL marks the jobs on a node that stopped responding), else the one running then | |

`<job>` is `job_id, user, job_name, partition`. It sees the jobs of whichever account runs it: if the cluster hides other users' jobs (`PrivateData=jobs`), run it as an account allowed to see every job to publish them all.

## Push the jobs

It needs only Python 3 and Slurm's `scontrol`, `sstat`, `sinfo` and `sacct` on `PATH` (or `SLURM_BIN`), so run it on any host with the Slurm client. `ctl.sh up` copies it to `$STATE_DIR/slurm-exporter/exporter.py`, readable by `$STATE_DIR`'s group; any copy works.

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

## Find why a node went down

Do it by hand with the same steps the exporter takes: the time Slurm set the reason, then the job records on the node at that time, requeued ones included (`-D`).
```bash
sinfo -N -n <node> -o "%N %T %E %H"     # state, reason, when it was set
sacct -a -D -X -N <node> -S <time - 2 min> -E <time + 1 min> -o JobID,User,JobName%40,State,Start,End
```
A `REQUEUED` record ending at that time is the job whose prolog failed; `NODE_FAIL` marks the jobs on a node that stopped responding. A Prolog error is Bright's pre-job health check failing; what it found is in `/cm/local/apps/slurm/var/pre-post-job-logs/failed-pre-jobs/job.<job>.log` on the node (readable by anyone; [gpu_processes.py](../prometheus/gpu_processes.py) pushes it). A `dmesg` finding such as `task python:<pid> blocked for more than 122 seconds` comes from a job running then, not from the requeued one.

## Notes

- `cpu` is Slurm's `CPU_IDs`, which are the OS's core numbers where the two numberings agree (they do when a job's `Cpus_allowed_list` in `/proc/<pid>/status` matches its `CPU_IDs`). `sstat` reports memory as Slurm last sampled it (`JobAcctGatherFrequency`). CPU use comes from node_exporter's per-core counters instead of `sstat`'s cumulative CPU time: a rate per core, live, and no extra call per job.
- Each push replaces the whole group `job="slurm_jobs"` (`PUSHGATEWAY_JOB` to push elsewhere, e.g. for a test), and a clean stop (SIGTERM from `timeout`, Ctrl-C) deletes it. A pusher that is killed leaves its last push in place; filter on `push_time_seconds` to ignore an old one.
- Without the token, or with another method or path, the push route doesn't match and the request gets a 404.
- What it pushes holds every visible job's ID, user and name, readable by everyone with Grafana access; on a cluster that keeps jobs private, push others' jobs only with the cluster admins' agreement.
