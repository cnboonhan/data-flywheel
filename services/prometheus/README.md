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
| `pushgateway` | `pushgateway:9091`, fed through Caddy's `/pushgateway` route (bearer `PUSHGATEWAY_TOKEN`) | what is pushed from outside, e.g. `slurm_job_gpu` and `slurm_job_cpu` (the GPUs and CPU cores each running job holds), `slurm_job_cpus`, `slurm_job_memory_bytes`, `slurm_job_rss_bytes` ([slurm-exporter/](../slurm-exporter/README.md)) |
| `pushgateway` (job `gpu_processes`) | `gpu_processes.py`, every minute | `gpu_process_memory_bytes` and `gpu_process_count` per node, GPU, user, workload and Slurm job; `node_cpu_memory_total_bytes`, `node_cpu_memory_available_bytes`; `gpu_processes_node_up` |

Every node series carries `instance` (the node's name), `partition` (its Slurm partition) and, for GPUs, `gpu` (its index).

## Use

1. Open Grafana, then **Dashboards → Flywheel → Cluster usage**, and pick a partition or node at the top:
   - **Now**: GPUs in use, free, not visible (nodes the collector can't reach) and held but idle; average GPU, CPU and CPU-memory use.
   - **GPU map**: each node's GPUs, their 15-min utilisation and who is on each.
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

## Notes

- The exporters run on the nodes already (the cluster installs them); this service only collects. `nodes.json` lists them by IP because containers on the service node can't resolve the cluster's host names.
- Without a pusher run by an account that sees every job, `slurm_job_gpu` has only some jobs or none. Without anyone pushing, a cluster admin can instead enable dcgm-exporter's job mapping (`--hpc-job-mapping-dir` with a Slurm prolog/epilog), which labels every GPU series with its job.
- Who is on each GPU comes from `gpu_processes.py`, which `ctl.sh monitor timer` runs every minute on `SERVICE_NODE` (`ctl.sh monitor` once): over ssh to every node, nvidia-smi's processes with their user and workload (Docker container, else Slurm job's script, else command), and each node's CPU memory. Your account must be able to ssh to the nodes; those it can't (the `xcyber` partition) show as not visible, attributed only through Slurm jobs the pusher sees.
- CPU memory is the NUMA nodes with CPUs only: on GB300 nodes the GPUs' memory is system memory too (node_exporter's `MemTotal` is ~2 TiB), and DCGM's `FB_USED` counts file cache in it, so GPU memory per workload comes from the processes instead.
- The dashboards are JSON in `../grafana/provisioning/dashboards/` ; edit the JSON (changes made in the UI are not kept), then restart Grafana.
