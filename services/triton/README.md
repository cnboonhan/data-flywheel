# triton

Triton Inference Server for image models and robot policies in the MLflow registry (now: `Qwen-Image-Edit-2511`), as the Gitea repo `triton`, a Triton model repository, lists them, at `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC, no token). It has `TRITON_NODE` to itself; Kimi K3 is on [Dynamo](../dynamo/README.md) (OpenAI API).

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

1. Register the weights in MLflow (a training run does; `download-models-hf` for Hugging Face models, by an admin). A π0.5 checkpoint is registered from a run with params `model=Pi_05`, `openpi_config` (a config [`r1pro.py`](models/Pi05.r1pro.picking_up_cup/1/r1pro.py) registers) and optionally `prompt`, its artifacts the step directory's `params/` and `assets/` (not `train_state/`):
   ```python
   with mlflow.start_run() as run:
       mlflow.log_params({"model": "Pi_05", "openpi_config": "pi05_r1pro_picking_up_cup", "prompt": "pick up the cup"})
       for d in ("params", "assets"):
           mlflow.log_artifacts(f"{step_dir}/{d}", f"checkpoint/{d}")
   mlflow.register_model(f"runs:/{run.info.run_id}/checkpoint", "Pi05.r1pro.picking_up_cup")
   ```
2. Add the model to the Gitea repo `triton` (`https://$SERVICE_HOST:$CADDY_PORT/gitea/admin/triton`), which is a [Triton model repository](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html) without the weights: a directory named as clients will call the model, with Triton's [`config.pbtxt`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html) and a version directory `1/` holding the backend's code (or, for vLLM, `model.json`). Name the weights in the config, and push to `main`; remove the directory to unload it. Copy a model like it from [`models/`](models/) (the seed).
   ```
   parameters { key: "mlflow_model" value { string_value: "Pi05.r1pro.picking_up_cup" } }
   parameters { key: "mlflow_version" value { string_value: "1" } }
   ```
3. The push runs the Gitea workflow `sync-triton` in the admin's `pipelines` repo, through the repo's webhook (`gitea_ci/sync-triton.yml`; an admin can dispatch it by hand with `force` or `dry_run`, [how](../gitea/README.md)): it copies what changed into `s3://triton/models/`, adding the parameter `weights` (the version's artifacts, read in place under `/mlflow`), loads it and checks it answers. The outcome is in the run's log and `s3://triton/sync/status.json` (`ready` or `failed`, with the error).

**Notes**
- **Who may deploy:** the admins and the deployer accounts, as for [Dynamo](../dynamo/README.md#notes).
- Runs on the three GPUs `TRITON_GPU_A`..`C` of `TRITON_NODE` (all shared; set in `services/.env`): a model with `instance_group [ { kind: KIND_GPU } ]` gets an instance on each, `gpus: [ N ]` pins it to one; a vLLM model takes one whole GPU (`TRITON_VLLM_GPU`; split across GPUs, a 122B model hit NVLink errors). Keep GPU-heavy Slurm jobs off the node (`--exclude=$TRITON_NODE`).
- Loading, unloading and settings aren't reachable from outside: `sync-triton` does them, and Caddy adds the key Triton requires.
- **Backend code** lives in each model's version directory in the repo (a Python backend's `1/model.py` and what it imports); a pushed change to a model's files reloads it. It runs inside Triton, so whoever can push to the repo runs code there. A model whose weights must sit in its version directory (ONNX, TensorRT plans) isn't supported: weights are always read in place from MLflow (the registry `DEPLOY_MLFLOW_*` in [`.env`](../.env.example) names, the stack's by default).
- After editing `Dockerfile`, rebuild on the node: `services/ctl.sh triton build`, then recreate the server: dispatch the Gitea workflow `triton-up` (`gitea_ci/triton-up.yml`; input `dry_run` shows what it would recreate), or run `services/ctl.sh triton up -d`. The same after changing its GPUs or flags (`compose.yml`, `.env`). Logs: `services/ctl.sh triton logs -f`. Then copy the image to `SERVICE_NODE`, where the runner starts the `sync-triton` job in it (13 GB, a few minutes):
  ```bash
  ssh $TRITON_NODE "bash -lc 'docker save flywheel-triton:26.08 | ssh $SERVICE_NODE \"bash -lc \\\"docker load\\\"\"'"
  ```
