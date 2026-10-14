# prometheus

Metrics store (config `prometheus.yml`) for every GPU node of the cluster and for Triton, kept 30 days in `$STATE_DIR/prometheus`. No UI or route of its own: read it in [Grafana](../grafana/README.md), dashboard **Flywheel → GPU nodes**.

| Job | Source | What |
|---|---|---|
| `dcgm` | NVIDIA dcgm-exporter on every node, port 9400 | per GPU: utilisation (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`), memory (`DCGM_FI_DEV_FB_USED`), power, temperature, clocks, NVLink bandwidth, XID errors, remapped rows |
| `node` | node_exporter on every node, port 9100 | CPU, memory, disk, network |
| `triton` | `triton:8002` | inference counts and latency per model |
| `pushgateway` | `pushgateway:9091`, fed through Caddy's `/pushgateway` route (bearer `PUSHGATEWAY_TOKEN`) | what is pushed from outside, e.g. `slurm_job_gpu` and `slurm_job_cpu` (the GPUs and CPU cores each running job holds), `slurm_job_cpus`, `slurm_job_memory_bytes`, `slurm_job_rss_bytes` ([slurm-exporter/](../slurm-exporter/README.md)) |

Every node series carries `instance` (the node's name), `partition` (its Slurm partition) and, for GPUs, `gpu` (its index).

## Use

1. Open Grafana, then **Dashboards → Flywheel → GPU nodes**, and pick a partition or node at the top. Its **Jobs** row has each job's GPUs, VRAM, CPU and RAM against what it was allocated; its **Nodes** row each node's CPU, RAM, GPU and VRAM use.
2. Or query in **Explore** (data source Prometheus), e.g. the GPUs idle for the last hour:
   ```
   avg_over_time(DCGM_FI_DEV_GPU_UTIL[1h]) == 0
   ```
3. Join GPU metrics to jobs through `slurm_job_gpu`, e.g. utilisation per job:
   ```
   avg by (job_id, user, job_name) (DCGM_FI_DEV_GPU_UTIL * on (instance, gpu) group_left (job_id, user, job_name) slurm_job_gpu)
   ```
   or CPU cores busy per job:
   ```
   sum by (job_id, user) (rate(node_cpu_seconds_total{mode!="idle"}[2m]) * on (instance, cpu) group_left (job_id, user) slurm_job_cpu)
   ```
   They hold the jobs the pusher's account can see; check the push is current with:
   ```
   time() - push_time_seconds{job="slurm_jobs"}
   ```
4. After nodes are added or removed, rewrite the node list on a login node; Prometheus picks it up by itself.
   ```bash
   services/prometheus/nodes.sh
   ```

## Notes

- The exporters run on the nodes already (the cluster installs them); this service only collects. `nodes.json` lists them by IP because containers on the service node can't resolve the cluster's host names.
- Without a pusher run by an account that sees every job, `slurm_job_gpu` has only some jobs or none. Without anyone pushing, a cluster admin can instead enable dcgm-exporter's job mapping (`--hpc-job-mapping-dir` with a Slurm prolog/epilog), which labels every GPU series with its job.
- The dashboard is adapted from NVIDIA dcgm-exporter's (Apache-2.0), in `../grafana/provisioning/dashboards/gpu-nodes.json`; edit the JSON, since changes made in the UI are not kept.
