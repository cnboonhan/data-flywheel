# triton

Triton Inference Server on all four GPUs of the service node, serving models from the MLflow registry at `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC, header `Authorization: Bearer $TRITON_TOKEN`).

## Serve a model

1. Set the alias `triton` on the registered model version to serve; remove it to unload.
   ```python
   mlflow.MlflowClient().set_registered_model_alias("ACT.arx_x5.RoboDojo.stack_bowls", "triton", "3")
   mlflow.MlflowClient().delete_registered_model_alias("ACT.arx_x5.RoboDojo.stack_bowls", "triton")
   ```
2. Wait for the Gitea workflow `sync-triton` (every 10 min), or dispatch it ([how](../gitea/README.md)). It packages the version into `s3://triton/models/<name>/<version>/`, loads it and runs a test inference.
3. Check the outcome in [Tracking](#tracking).

## Call a model

Use `TRITON_URL` and `TRITON_TOKEN` from `slurm.env`:

```python
import os, tritonclient.grpc as tg
c = tg.InferenceServerClient(os.environ["TRITON_URL"], ssl=True, root_certificates=os.environ["AWS_CA_BUNDLE"])
chunk = c.infer(name, inputs, headers={"authorization": "Bearer " + os.environ["TRITON_TOKEN"]}).as_numpy("action")
```

Models are stateless: one observation in, one action chunk out. `state` is packed per arm as `[arm, ee]` in the embodiment's `arms` order.

| Run param `model` | Backend | Inputs | Output |
|---|---|---|---|
| `ACT` | Python (`backends/xpolicylab_act.py`) | `state` FP32 `[B, D]`, one UINT8 `[B, H, W, 3]` RGB per camera | `action` `[B, chunk, D]` |
| `MLP` | ONNX Runtime | `state` FP32 `[B, D]` | `action` `[B, 1, D]` |

Other models are tagged `unsupported`; to add one, write a `package_<model>` function in `sync.py`.

## Tracking

| Where | What |
|---|---|
| MLflow version tags | `triton.status` (`ready`, `failed`, `unsupported`, `unloaded`), `triton.error`, `triton.synced_at` |
| `GET /v2/models/<name>/config` → `parameters` | MLflow model, version, run, embodiment, mix, data fingerprint, git commit |
| `s3://triton/sync/status.json`, `sync/log/<time>.json` | current state; one file per change |
| `sync.py --status`, `sync-triton` run log | Triton's loaded models with their MLflow source |

## Notes

- C03 is still a Slurm node, so Slurm jobs placed there share its GPUs with Triton. Until the admin removes C03's GPUs from Slurm, submit GPU-heavy jobs with `--exclude=C2-GB300-02-C03`.
- Send gRPC metadata keys in lowercase (`authorization`). `tritonclient.http` adds entries to the `headers` dict it is given, so pass a fresh dict on every call.
- Expect about 40 ms per ACT call with three 640×480 cameras from the login node.
- After editing `Dockerfile` (`tritonserver:26.08-py3` plus torch 2.14 cu130; CUDA 13 runs in forward-compatibility mode on driver 580, which sm_103 needs), rebuild with `services/ctl.sh up --build`.
- `sync.py` is the reconciler. It runs in the same image with the checkout mounted, so a change to it takes effect on the next sync without a rebuild. After changing `backends/` or a packager, dispatch `sync-triton` with `force` set to the affected models to repackage them.
- The server reads the bucket directory read-only and reloads every model on restart. Leave `models/.keep` in place: the gateway deletes an empty folder, and Triton loses its repository with it.
- Metrics are on port 8002 inside the compose network; nothing scrapes them.
