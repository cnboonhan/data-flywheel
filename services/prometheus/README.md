# prometheus

Metrics store (config `prometheus.yml`) for every GPU node of the cluster and for Triton, kept 30 days in `$STATE_DIR/prometheus`. No UI or route of its own: read it in [Grafana](../grafana/README.md), dashboard **Flywheel → GPU nodes**.

| Job | Source | What |
|---|---|---|
| `dcgm` | NVIDIA dcgm-exporter on every node, port 9400 | per GPU: utilisation (`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_PROF_GR_ENGINE_ACTIVE`, `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`), memory (`DCGM_FI_DEV_FB_USED`), power, temperature, clocks, NVLink bandwidth, XID errors, remapped rows |
| `node` | node_exporter on every node, port 9100 | CPU, memory, disk, network |
| `triton` | `triton:8002` | inference counts and latency per model |
| `slurm` | `slurm-exporter:9105` ([slurm-exporter/](../slurm-exporter/)) | `slurm_job_gpu{instance, gpu, job_id, user, job_name, partition}`: the GPUs each running job holds |

Every node series carries `instance` (the node's name), `partition` (its Slurm partition) and, for GPUs, `gpu` (its index).

## Use

1. Open Grafana, then **Dashboards → Flywheel → GPU nodes**, and pick a partition or node at the top.
2. Or query in **Explore** (data source Prometheus), e.g. the GPUs idle for the last hour:
   ```
   avg_over_time(DCGM_FI_DEV_GPU_UTIL[1h]) == 0
   ```
3. Join GPU metrics to jobs through `slurm_job_gpu`, e.g. utilisation per job:
   ```
   avg by (job_id, user, job_name) (DCGM_FI_DEV_GPU_UTIL * on (instance, gpu) group_left (job_id, user, job_name) slurm_job_gpu)
   ```
4. After nodes are added or removed, rewrite the node list on a login node; Prometheus picks it up by itself.
   ```bash
   services/prometheus/nodes.sh
   ```

## Notes

- The exporters run on the nodes already (the cluster installs them); this service only collects. `nodes.json` lists them by IP because containers on the service node can't resolve the cluster's host names.
- `slurm_job_gpu` covers only the jobs the stack's account (`SERVICE_UID`) can see: the cluster hides other users' jobs (`PrivateData=jobs`). For everyone's, a cluster admin enables dcgm-exporter's job mapping (`--hpc-job-mapping-dir` with a Slurm prolog/epilog), which labels every GPU series with its job.
- The dashboard is adapted from NVIDIA dcgm-exporter's (Apache-2.0), in `../grafana/provisioning/dashboards/gpu-nodes.json`; edit the JSON, since changes made in the UI are not kept.
