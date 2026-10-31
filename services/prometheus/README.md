# prometheus

Metrics store (config `prometheus.yml`) for every GPU node of the cluster and for the served models (Triton, Dynamo), kept 90 days (at most 250 GB) in `$STATE_DIR/prometheus`. No UI or route of its own: read it in [Grafana](../grafana/README.md), dashboards **Flywheel → Cluster usage** and **Model usage**.

| Job | Source | What |
|---|---|---|
| `dcgm` | NVIDIA dcgm-exporter on every node, port 9400 | per GPU: utilisation (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`), memory (`DCGM_FI_DEV_FB_USED`), power, temperature, clocks, NVLink bandwidth, XID errors, remapped rows |
| `node` | node_exporter on every node, port 9100 | CPU, memory, disk, network |
| `caddy` | `caddy:9180` | every request through the stack's port, per site: rate, latency, requests in flight, errors, Caddy's CPU (query in Explore, e.g. `caddy_http_requests_total`) |
| `triton` | Triton's metrics port on `TRITON_NODE` | per model: inference counts and latency; for vLLM-backend models, vLLM's tokens and time to first token |
| `llm-gateway` | the gateway (`DYNAMO_HEAD:8199`) | requests per model and class (normal, critical), in flight, waiting for a switch, the awake model, switches |
| `dynamo` | each Dynamo model's frontend and engine, relayed by the gateway (`/metrics/<model>/frontend\|engine`; labels `model`, `part`) | vLLM's tokens, time to first token, running and waiting requests, KV cache (`vllm:*`); Dynamo's request and latency metrics (`dynamo_*`) |
| `pushgateway` | `pushgateway:9091`, fed through Caddy's `/pushgateway` route (bearer `PUSHGATEWAY_TOKEN`) | what is pushed from outside, e.g. `slurm_job_gpu` and `slurm_job_cpu` (the GPUs and CPU cores each running job holds), `slurm_job_cpus`, `slurm_job_memory_bytes`, `slurm_job_rss_bytes` ([Slurm jobs](#slurm-jobs)) |
| `pushgateway` (job `gpu_processes`) | `gpu_processes.py`, every minute | `gpu_process_memory_bytes` and `gpu_process_count` per node, GPU, user, workload and Slurm job; `node_cpu_memory_total_bytes`, `node_cpu_memory_available_bytes`; `gpu_processes_node_up` and `gpu_processes_node_status` (ok, timeout, ssh_denied: why a node couldn't be read); `node_mount_ok{path}` (the shared file systems in `gpu_processes.py`'s `MOUNTS`; 0: jobs that use it fail there); `node_health_check_failure_seconds` (a node's latest failed pre-job health check, the Prolog error that drains it) |

Every node series carries `instance` (the node's name), `partition` (its Slurm partition) and, for GPUs, `gpu` (its index).

## Use

1. Open Grafana, then **Dashboards → Flywheel → Cluster usage**, and pick a partition or node at the top:
   - **Now**: GPUs in use, free, not visible (nodes the collector can't reach), held but idle, and holding memory with no job (red when any); average GPU, CPU and CPU-memory use.
   - **GPU map**: the GPUs holding memory while no running Slurm job holds them (work started outside Slurm, or left behind by a job that ended; the stack's own `flywheel-*` containers are left out), each node's GPUs, their 15-min utilisation and who is on each.
   - **Over time**: GPU, CPU and CPU-memory use by node; GPUs in use by user.
   - **Who is using them**: every workload on the GPUs (Slurm or not), Slurm jobs' allocation against use (click a job for **Slurm job**), GPU-hours by user.
   For the served models, open **Flywheel → Model usage**.
2. Or query in **Explore** (data source Prometheus), e.g. who is on the GPUs now, or the GPUs idle for the last hour:
   ```
   sum by (user, workload) (gpu_process_memory_bytes)
   avg_over_time(DCGM_FI_DEV_GPU_UTIL[1h]) == 0
   ```
3. After nodes are added or removed, rewrite the node list on a login node; Prometheus and the collector pick it up by themselves.
   ```bash
   services/prometheus/nodes.sh
   ```

## Slurm jobs

`slurm_exporter.py` pushes what each running Slurm job holds and uses, and each node's state and latest job, to the stack's Pushgateway, so Prometheus can join it with the nodes' metrics:

| Metric | What | Joins |
|---|---|---|
| `slurm_job_gpu{instance, gpu, <job>}` | each GPU the job holds | dcgm-exporter's GPU metrics, `on (instance, gpu)` |
| `slurm_job_cpu{instance, cpu, <job>}` | each CPU core the job is pinned to | node_exporter's `node_cpu_seconds_total`, `on (instance, cpu)` |
| `slurm_job_cpus{<job>}`, `slurm_job_memory_bytes{<job>}` | CPUs and memory allocated (0 bytes: none requested) | |
| `slurm_job_rss_bytes{<job>}` | memory in use, summed over its running steps (`sstat`) | |
| `slurm_job_pending_seconds{<job>, reason}`, `slurm_job_pending_gpus`, `_nodes`, `_priority` | the queue: each pending job, how long it has waited, why (`scontrol`'s reason), what it asks for, its priority (0: held) | |
| `slurm_node_state{instance, state, reason, partition}` | each node's state (`sinfo`: idle, mixed, allocated, drained, down, ...; a trailing `*`: not responding) and why | node metrics, `on (instance)` |
| `slurm_node_last_job_start_seconds{instance, <job>, job_state}`, `..._end_seconds` | the latest job started on the node (`sacct`, last 30 days) and when it started and ended (0 while it runs) | |
| `slurm_node_reason_seconds{instance}` | when Slurm set a down or drained node's reason (`sinfo %H`) | |
| `slurm_node_cause_job_seconds{instance, cause_job, cause_user, cause_name, cause_state}` | the job it went down on: the one Slurm requeued or failed on it then (`sacct -D`: a Prolog error requeues the job it was starting, NODE_FAIL marks the jobs on a node that stopped responding), else the one running then | |

`<job>` is `job_id, user, job_name, partition`. It sees the jobs of whichever account runs it: if the cluster hides other users' jobs (`PrivateData=jobs`), run it as an account allowed to see every job to publish them all.

### Push the jobs

It needs only Python 3 and Slurm's `scontrol`, `sstat`, `sinfo` and `sacct` on `PATH` (or `SLURM_BIN`), so run it on any host with the Slurm client. `ctl.sh up` copies it to `$STATE_DIR/slurm-exporter/exporter.py` (that name, so existing crontabs keep working), readable by `$STATE_DIR`'s group; any copy works.

1. Get `PUSHGATEWAY_TOKEN` from the admin (`services/.env`) and the stack's CA certificate ([quickstart](../../quickstart.md) step 3).
2. Check it pushes (stop it with Ctrl-C; that also deletes what it pushed).
   ```bash
   export PUSHGATEWAY_TOKEN=<token> PUSHGATEWAY_CA=<path to the CA certificate>
   python3 slurm_exporter.py --push https://$SERVICE_HOST:$CADDY_PORT/pushgateway
   ```
3. Keep it running with Slurm's cron, so it survives logouts (`scrontab -e`; it restarts every 30 minutes and pushes every 30 s). Put the token in a file only you can read rather than in the crontab:
   ```
   */30 * * * * . ~/.pushgateway.env && timeout 1790 python3 <STATE_DIR>/slurm-exporter/exporter.py --push https://<SERVICE_HOST>:<CADDY_PORT>/pushgateway
   ```
4. Check that Prometheus gets it: `time() - push_time_seconds{job="slurm_jobs"}` stays under 60 and `slurm_job_gpu` lists the running jobs.

### Find why a node went down

Do it by hand with the same steps the exporter takes: the time Slurm set the reason, then the job records on the node at that time, requeued ones included (`-D`).
```bash
sinfo -N -n <node> -o "%N %T %E %H"     # state, reason, when it was set
sacct -a -D -X -N <node> -S <time - 2 min> -E <time + 1 min> -o JobID,User,JobName%40,State,Start,End
```
A `REQUEUED` record ending at that time is the job whose prolog failed; `NODE_FAIL` marks the jobs on a node that stopped responding. A Prolog error is Bright's pre-job health check failing; what it found is in `/cm/local/apps/slurm/var/pre-post-job-logs/failed-pre-jobs/job.<job>.log` on the node (readable by anyone; [gpu_processes.py](gpu_processes.py) pushes it). A `dmesg` finding such as `task python:<pid> blocked for more than 122 seconds` comes from a job running then, not from the requeued one.

**Notes**

- `cpu` is Slurm's `CPU_IDs`, which are the OS's core numbers where the two numberings agree (they do when a job's `Cpus_allowed_list` in `/proc/<pid>/status` matches its `CPU_IDs`). `sstat` reports memory as Slurm last sampled it (`JobAcctGatherFrequency`). CPU use comes from node_exporter's per-core counters instead of `sstat`'s cumulative CPU time: a rate per core, live, and no extra call per job.
- Each push replaces the whole group `job="slurm_jobs"` (`PUSHGATEWAY_JOB` to push elsewhere, e.g. for a test), and a clean stop (SIGTERM from `timeout`, Ctrl-C) deletes it. A pusher that is killed leaves its last push in place; filter on `push_time_seconds` to ignore an old one.
- Without the token, or with another method or path, the push route doesn't match and the request gets a 404.
- What it pushes holds every visible job's ID, user and name, readable by everyone with Grafana access; on a cluster that keeps jobs private, push others' jobs only with the cluster admins' agreement.

## Notes

- The exporters run on the nodes already (the cluster installs them); this service only collects. `nodes.json` lists them by IP because containers on the service node can't resolve the cluster's host names.
- Without a pusher run by an account that sees every job, `slurm_job_gpu` has only some jobs or none. Without anyone pushing, a cluster admin can instead enable dcgm-exporter's job mapping (`--hpc-job-mapping-dir` with a Slurm prolog/epilog), which labels every GPU series with its job.
- Who is on each GPU comes from `gpu_processes.py`, which a timer `ctl.sh up` installs runs every minute on `SERVICE_NODE` (`ctl.sh monitor` once): over ssh to every node, nvidia-smi's processes with their user and workload (Docker container, else Slurm job's script, else command), and each node's CPU memory. Your account must be able to ssh to the nodes; those it can't show as not visible, attributed only through Slurm jobs the pusher sees.
- CPU memory is the NUMA nodes with CPUs only: on nodes whose GPU memory is system memory too (GB300: node_exporter's `MemTotal` is ~2 TiB), and DCGM's `FB_USED` counts file cache in it, so GPU memory per workload comes from the processes instead.
- The dashboards are JSON in [`../grafana/provisioning/dashboards/`](../grafana/provisioning/dashboards/); edit the JSON (changes made in the UI are not kept), then restart Grafana.
