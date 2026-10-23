# triton

Triton Inference Server for the image models, robot policies and one language model (`qwen3.5-122b-a10b`, on vLLM) in the MLflow registry, at `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC, no token). It has `TRITON_NODE` to itself; Kimi K3 is on [Dynamo](../dynamo/README.md) (OpenAI API).

## Example usage

Set `TRITON_URL` and the CA as in the [quickstart](../../quickstart.md) (on the cluster: `flywheel.env`).

```bash
curl --cacert $FLYWHEEL_CA https://$TRITON_URL/v2/health/ready -o /dev/null -w '%{http_code}\n'   # 200: up
curl --cacert $FLYWHEEL_CA -X POST https://$TRITON_URL/v2/repository/index                     # models and state
curl --cacert $FLYWHEEL_CA https://$TRITON_URL/v2/models/moge-3-vitg                          # its inputs and outputs
curl --cacert $FLYWHEEL_CA https://$TRITON_URL/v2/models/Qwen3.5-122B-A10B/generate \
  -d '{"text_input": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", "sampling_parameters": "{\"max_tokens\": 64}"}'   # the language model
```

Call a model from Python through `triton_endpoint.py` (`services/triton` on `sys.path`; `pip install tritonclient[http] numpy`): one `Endpoint(role, model)` per model, which `<ROLE>_ENDPOINT` can point at another model or Triton.

```python
import numpy as np
from triton_endpoint import Endpoint

jpg = open("view.jpg", "rb").read()
depth = Endpoint("DEPTH", "moge-3-vitg")
distance = depth.infer({"image": [jpg], "fov_x": np.array([90.0], np.float32)}, ["distance"])["distance"]   # metres, [H, W]

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
| `moge-3-vitg` | `image` BYTES, `fov_x` FP32 (degrees; 0: equirectangular panorama); optional `view_size` (1024), `merge_width` (0: full) | `distance` FP32 `[H, W]` (m), `mask` BOOL |
| `llmdet_large` | `image` BYTES, `labels` BYTES `[N]`; optional `box_threshold` (0.35), `text_threshold` (0.25), `nms` (0.7) | `boxes` FP32 `[K, 4]` (px), `scores`, `names` |
| `sam2.1-hiera-large` | `image` BYTES, `boxes` FP32 `[K, 4]` (px) | `masks` BOOL `[K, H, W]` |
| `TripoSG` | `image` BYTES (RGBA cutout of one object); optional `seed` (0), `steps` (50), `guidance` (7.0), `faces` (-1: all) | `mesh` BYTES (GLB, shape only, y up, within ±1) |
| ACT policies | `state` FP32 `[B, D]`, one UINT8 `[B, H, W, 3]` per camera, named after it | `action` FP32 `[B, chunk, D]` |
| MLP policies | `state` FP32 `[B, D]` | `action` FP32 `[B, 1, D]` |
| `Qwen3.5-122B-A10B` (vLLM) | `text_input` (chat template applied), optional `image` (base64), `sampling_parameters` (JSON string) | `text_output` |

Expect an image edit at 1024×1024 in ~16 s, MoGe in ~0.6 s per 1024 px view (a 4096×2048 panorama ~110 s), LLMDet and SAM under a second, TripoSG ~12 s, an ACT call ~40 ms.

## Serve a model

1. Set the alias `triton` on the version to serve; remove it to unload.
   ```python
   mlflow.MlflowClient().set_registered_model_alias("ACT.arx_x5.RoboDojo.stack_bowls", "triton", "3")
   ```
2. Dispatch the Gitea workflow `sync-triton` (`gitea_ci/sync-triton.yml`; [how](../gitea/README.md)): it packages, loads and test-runs the version, and writes the outcome to the version's tags (`triton.status`: `ready`, `failed`, `unsupported`, `unloaded`; `triton.error`).

**Notes**
- Runs on two GPUs of `TRITON_NODE`: `TRITON_GPU` for the image models and policies, `TRITON_LLM_GPU` for the 122B (a whole GPU: split across GPUs it hit NVLink errors). GPU 0 there isn't ours; keep GPU-heavy Slurm jobs off the node (`--exclude=$TRITON_NODE`).
- Loading, unloading and settings aren't reachable from outside: `sync-triton` does them, and Caddy adds the key Triton requires.
- A new model type needs a `package_<model>` in `sync.py` or a backend in `backends/` (inputs and outputs in `PYTHON_IO`); others are tagged `unsupported`. After changing a backend, dispatch `sync-triton` with `force` set to the models it serves.
- After editing `Dockerfile`, rebuild on the node: `services/ctl.sh triton build && services/ctl.sh triton up -d`. Logs: `services/ctl.sh triton logs -f`.
- TripoSG skips background removal (its RMBG-1.4 is non-commercial): give it a cutout. Its yaw is close to the input's view but not exact, so fit it against the input.
