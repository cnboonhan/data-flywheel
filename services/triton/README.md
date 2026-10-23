# triton

Triton Inference Server for image models and robot policies in the MLflow registry (now: `Qwen-Image-Edit-2511`), as the Gitea repo `triton`'s `models.yaml` lists them, at `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC, no token). It has `TRITON_NODE` to itself; Kimi K3 is on [Dynamo](../dynamo/README.md) (OpenAI API).

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

policy = Endpoint("POLICY", "Pi05.r1pro.picking_up_cup")   # a π0.5 policy: one observation in, an action chunk out
action = policy.infer({"state": state.astype(np.float32),                                 # [D], the trained joint vector
                       "cam_high": head, "cam_left_wrist": left, "cam_right_wrist": right},  # uint8 [H, W, 3]
                      ["action"])["action"]                                                # [chunk, D]
```

| Model | Inputs | Output |
|---|---|---|
| `Qwen-Image-Edit-2511` | `images` BYTES `[1-3]` (picture 1 first), `prompt` BYTES; optional `negative_prompt`, `steps` (40), `seed` (0), `true_cfg_scale` (4.0) | `image` BYTES (PNG) |
| π0.5 policies | `state` FP32 `[D]`, `cam_high`, `cam_left_wrist`, `cam_right_wrist` UINT8 `[H, W, 3]`; optional `prompt` BYTES (default: the run's) | `action` FP32 `[chunk, D]` (absolute joint targets) |
| MLP policies | `state` FP32 `[B, D]` | `action` FP32 `[B, 1, D]` |

Expect an image edit at 1024×1024 in ~16 s.

## Serve a model

1. Register the weights in MLflow (a training run does; `download-models-hf` for Hugging Face models, by an admin). A π0.5 checkpoint is registered from a run with params `model=Pi_05`, `openpi_config` (a config [`r1pro.py`](../../sensors/autolearn/adapter/pi05/r1pro.py) registers) and optionally `prompt`, its artifacts the step directory's `params/` and `assets/` (not `train_state/`):
   ```python
   with mlflow.start_run() as run:
       mlflow.log_params({"model": "Pi_05", "openpi_config": "pi05_r1pro_picking_up_cup", "prompt": "pick up the cup"})
       for d in ("params", "assets"):
           mlflow.log_artifacts(f"{step_dir}/{d}", f"checkpoint/{d}")
   mlflow.register_model(f"runs:/{run.info.run_id}/checkpoint", "Pi05.r1pro.picking_up_cup")
   ```
2. List the version in the spec, `models.yaml` in the Gitea repo `triton` (`https://$SERVICE_HOST:$CADDY_PORT/gitea/admin/triton`), and push to `main`; remove an entry to unload it. Optional per model: `backend`, `gpu`, `vllm` (engine arguments); [`models.yaml`](models.yaml) explains them.
   ```yaml
   models:
     - {name: Pi05.r1pro.picking_up_cup, version: 1}
   ```
3. The push runs the Gitea workflow `sync-triton` in the admin's `pipelines` repo, through the repo's webhook (`gitea_ci/sync-triton.yml`; an admin can dispatch it by hand with `force` or `dry_run`, [how](../gitea/README.md)): it packages, loads and test-runs what changed. The outcome is in the run's log and `s3://triton/sync/status.json` (`ready`, `failed`, `unsupported`, with the error).

**Notes**
- **Who may deploy:** the admins, and the accounts `services/ctl.sh deployer add <name>` makes (Gitea only, no Keycloak, a random password printed once): they may push the repos `dynamo` and `triton` and nothing else, so they choose which registered versions are served and with which flags, but registering weights in MLflow stays an admin's job. The repos run no workflows themselves; their webhooks start the admin's syncs (`services/ctl.sh specs` makes repos, webhooks and tokens).
- Runs on three GPUs of `TRITON_NODE` (`TRITON_GPU_A`..`C`, all shared): an image model gets an instance on each, unless the spec pins it to one with `gpu`; a vLLM model takes one whole GPU (`TRITON_VLLM_GPU`; split across GPUs the 122B hit NVLink errors). GPU 0 there isn't ours; keep GPU-heavy Slurm jobs off the node (`--exclude=$TRITON_NODE`).
- Loading, unloading and settings aren't reachable from outside: `sync-triton` does them, and Caddy adds the key Triton requires.
- **Backends** are edited in the Gitea repo `triton`, `backends/<name>.py` (the `backends/` here are the seed): a Python backend declares its inputs, outputs, smoke-test request and Hugging Face repos as `TRITON_*` literals at its top (see `qwen_image_edit.py`), which `sync.py` reads without running the file. A pushed change to a backend repackages and reloads the models that use it. Backend code runs inside Triton, so whoever can push to the repo runs code there. A model type with no backend needs a `package_<model>` in `sync.py`; others are reported `unsupported`.
- After editing `Dockerfile`, rebuild on the node: `services/ctl.sh triton build`, then recreate the server: dispatch the Gitea workflow `triton-up` (`gitea_ci/triton-up.yml`; input `dry_run` shows what it would recreate), or run `services/ctl.sh triton up -d`. The same after changing its GPUs or flags (`compose.yml`, `.env`). Logs: `services/ctl.sh triton logs -f`. Then copy the image to `SERVICE_NODE`, where the runner starts the `sync-triton` job in it (13 GB, a few minutes):
  ```bash
  ssh $TRITON_NODE "bash -lc 'docker save flywheel-triton:26.08 | ssh $SERVICE_NODE \"bash -lc \\\"docker load\\\"\"'"
  ```
