# triton

Triton Inference Server for image models and robot policies in the MLflow registry (now: `Qwen-Image-Edit-2511`), as the Gitea `pipelines` repo's `triton/models.yaml` lists them, at `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC, no token). It has `TRITON_NODE` to itself; Kimi K3 is on [Dynamo](../dynamo/README.md) (OpenAI API).

## Example usage

Set `TRITON_URL` and the CA as in the [quickstart](../../quickstart.md) (on the cluster: `flywheel.env`).

```bash
curl --cacert $FLYWHEEL_CA https://$TRITON_URL/v2/health/ready -o /dev/null -w '%{http_code}\n'   # 200: up
curl --cacert $FLYWHEEL_CA -X POST https://$TRITON_URL/v2/repository/index                     # models and state
curl --cacert $FLYWHEEL_CA https://$TRITON_URL/v2/models/Qwen-Image-Edit-2511                 # its inputs and outputs
```

Call a model from Python through `triton_endpoint.py` (`services/triton` on `sys.path`; `pip install tritonclient[http] numpy`): one `Endpoint(role, model)` per model, which `<ROLE>_ENDPOINT` can point at another model or Triton.

```python
import numpy as np
from triton_endpoint import Endpoint

jpg = open("view.jpg", "rb").read()
edit = Endpoint("EDIT", "Qwen-Image-Edit-2511")
png = edit.infer({"images": [jpg], "prompt": [b"Remove the bucket."]}, ["image"])["image"][0]              # PNG bytes

policy = Endpoint("POLICY", "ACT.arx_x5.RoboDojo.stack_bowls")   # a trained policy: one observation in, an action chunk out
action = policy.infer({"state": state[None].astype(np.float32),                                     # [1, D], per arm [arm, ee]
                       **{cam: rgb[None] for cam, rgb in images.items()}},                          # cam_head, ...: uint8 [1, H, W, 3]
                      ["action"])["action"]                                                          # [1, chunk, D]
```

| Model | Inputs | Output |
|---|---|---|
| `Qwen-Image-Edit-2511` | `images` BYTES `[1-3]` (picture 1 first), `prompt` BYTES; optional `negative_prompt`, `steps` (40), `seed` (0), `true_cfg_scale` (4.0) | `image` BYTES (PNG) |
| ACT policies | `state` FP32 `[B, D]`, one UINT8 `[B, H, W, 3]` per camera, named after it | `action` FP32 `[B, chunk, D]` |
| MLP policies | `state` FP32 `[B, D]` | `action` FP32 `[B, 1, D]` |

Expect an image edit at 1024×1024 in ~16 s, an ACT call ~40 ms.

## Serve a model

1. Register the weights in MLflow (a training run does; `download-models-hf` for Hugging Face models, by an admin).
2. List the version in the spec, `triton/models.yaml` in the Gitea `pipelines` repo, and push to `main`; remove an entry to unload it. Optional per model: `backend`, `gpu`, `vllm` (engine arguments); [`models.yaml`](models.yaml) explains them.
   ```yaml
   models:
     - {name: ACT.arx_x5.RoboDojo.stack_bowls, version: 3}
   ```
3. The push runs the Gitea workflow `sync-triton` (`gitea_ci/sync-triton.yml`; dispatch it by hand with `force` or `dry_run`, [how](../gitea/README.md)): it packages, loads and test-runs what changed. The outcome is in the run's log and `s3://triton/sync/status.json` (`ready`, `failed`, `unsupported`, with the error).

**Notes**
- Runs on three GPUs of `TRITON_NODE` (`TRITON_GPU_A`..`C`, all shared): an image model gets an instance on each, unless the spec pins it to one with `gpu`; a vLLM model takes one whole GPU (`TRITON_VLLM_GPU`; split across GPUs the 122B hit NVLink errors). GPU 0 there isn't ours; keep GPU-heavy Slurm jobs off the node (`--exclude=$TRITON_NODE`).
- Loading, unloading and settings aren't reachable from outside: `sync-triton` does them, and Caddy adds the key Triton requires.
- **Backends** are edited in the Gitea `pipelines` repo, `triton/backends/<name>.py` (the `backends/` here are the seed): a Python backend declares its inputs, outputs, smoke-test request and Hugging Face repos as `TRITON_*` literals at its top (see `qwen_image_edit.py`), which `sync.py` reads without running the file. A pushed change to a backend repackages and reloads the models that use it. Backend code runs inside Triton, so whoever can push to the repo runs code there. A model type with no backend needs a `package_<model>` in `sync.py`; others are reported `unsupported`.
- After editing `Dockerfile`, rebuild on the node: `services/ctl.sh triton build`, then recreate the server: dispatch the Gitea workflow `triton-up` (`gitea_ci/triton-up.yml`; input `dry_run` shows what it would recreate), or run `services/ctl.sh triton up -d`. The same after changing its GPUs or flags (`compose.yml`, `.env`). Logs: `services/ctl.sh triton logs -f`. Then copy the image to `SERVICE_NODE`, where the runner starts the `sync-triton` job in it (13 GB, a few minutes):
  ```bash
  ssh $TRITON_NODE "bash -lc 'docker save flywheel-triton:26.08 | ssh $SERVICE_NODE \"bash -lc \\\"docker load\\\"\"'"
  ```
