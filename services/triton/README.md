# triton

Triton Inference Server on all four GPUs of the service node, serving models from the MLflow registry. Endpoint: `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC), header `Authorization: Bearer $TRITON_TOKEN`. `TRITON_URL` and `TRITON_TOKEN` are in `slurm.env`.

C03 is still a Slurm node: jobs placed there share its GPUs with Triton until the admin removes them from Slurm.

## Serve a model

Set the alias `triton` on a registered model version; remove it to unload. The Gitea workflow `sync-triton` (every 10 min, or dispatch) packages it into `s3://triton/models/<name>/<version>/`, loads it, and runs a test inference.

```python
mlflow.MlflowClient().set_registered_model_alias("ACT.arx_x5.RoboDojo.stack_bowls", "triton", "3")
```

## Tracking

| Where | What |
|---|---|
| MLflow version tags | `triton.status` (`ready`, `failed`, `unsupported`, `unloaded`), `triton.error`, `triton.synced_at` |
| `GET /v2/models/<name>/config` → `parameters` | MLflow model, version, run, embodiment, mix, data fingerprint, git commit |
| `s3://triton/sync/status.json`, `sync/log/<time>.json` | current state; one file per change |
| `sync.py --status`, `sync-triton` run log | Triton's loaded models with their MLflow source |

## Models

Stateless: one observation in, one action chunk out. `state` is packed per arm as `[arm, ee]` in the embodiment's `arms` order.

| Run param `model` | Backend | Inputs | Output |
|---|---|---|---|
| `ACT` | Python (`backends/xpolicylab_act.py`) | `state` FP32 `[B, D]`, one UINT8 `[B, H, W, 3]` RGB per camera | `action` `[B, chunk, D]` |
| `MLP` | ONNX Runtime | `state` FP32 `[B, D]` | `action` `[B, 1, D]` |

Other models are tagged `unsupported`; add a `package_<model>` function to `sync.py`.

## Client

```python
c = tritonclient.grpc.InferenceServerClient(os.environ["TRITON_URL"], ssl=True, root_certificates=os.environ["AWS_CA_BUNDLE"])
chunk = c.infer(name, inputs, headers={"authorization": "Bearer " + os.environ["TRITON_TOKEN"]}).as_numpy("action")
```

gRPC metadata keys must be lowercase. `tritonclient.http` modifies the `headers` dict, so pass a fresh one per call. ACT with three 640×480 cameras: 40 ms per call from the login node.

## Files

- `Dockerfile`: `tritonserver:26.08-py3` (CUDA 13 forward-compat on driver 580, needed for sm_103) + torch 2.14 cu130. Rebuild with `ctl.sh up --build`.
- `sync.py`: the reconciler; runs in the same image with the checkout mounted.
- The server reads the bucket directory read-only (`sync.py` keeps `models/.keep` so the folder never disappears) and reloads everything on restart. Metrics on port 8002 (not scraped).
