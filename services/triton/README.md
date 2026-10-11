# triton

Triton Inference Server on all GPUs of the service node (`SERVICE_NODE` in [`.env.example`](../.env.example)), serving models from the MLflow registry at `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC, header `Authorization: Bearer $TRITON_TOKEN`).

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

Call an LLM or VLM (vLLM backend) through the `generate` endpoint. Apply the model's chat template yourself and send images as base64:

```python
import base64, json, os, requests
from transformers import AutoProcessor
proc = AutoProcessor.from_pretrained("Qwen/Qwen3.5-122B-A10B")      # the tokenizer and chat template only
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is on the counter?"}]}]
prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
r = requests.post(f"https://{os.environ['TRITON_URL']}/v2/models/Qwen3.5-122B-A10B/generate",
                  headers={"Authorization": "Bearer " + os.environ["TRITON_TOKEN"]}, verify=os.environ["AWS_CA_BUNDLE"],
                  json={"text_input": prompt, "image": [base64.b64encode(open("view.jpg", "rb").read()).decode()],
                        "exclude_input_in_output": True,
                        "sampling_parameters": json.dumps({"temperature": 0, "max_tokens": 1000})})
answer = r.json()["text_output"]
```

Send many requests at once: vLLM batches them on the GPU.

| Run param `model` | Backend | Inputs | Output |
|---|---|---|---|
| `ACT` | Python (`backends/xpolicylab_act.py`) | `state` FP32 `[B, D]`, one UINT8 `[B, H, W, 3]` RGB per camera | `action` `[B, chunk, D]` |
| `MLP` | ONNX Runtime | `state` FP32 `[B, D]` | `action` `[B, 1, D]` |
| Hugging Face model (`download-models-hf`) | vLLM | `text_input`, optional `image` (base64), `sampling_parameters` (JSON) | `text_output` |

Other models are tagged `unsupported`; to add one, write a `package_<model>` function in `sync.py`.

## Tracking

| Where | What |
|---|---|
| MLflow version tags | `triton.status` (`ready`, `failed`, `unsupported`, `unloaded`), `triton.error`, `triton.synced_at` |
| `GET /v2/models/<name>/config` → `parameters` | MLflow model, version, run, embodiment, mix, data fingerprint, git commit |
| `s3://triton/sync/status.json`, `sync/log/<time>.json` | current state; one file per change |
| `sync.py --status`, `sync-triton` run log | Triton's loaded models with their MLflow source |

## Notes

- If the service node is also a Slurm node, jobs placed there share its GPUs with Triton. Until its GPUs are removed from Slurm, submit GPU-heavy jobs with `--exclude=$SERVICE_NODE` (`slurm.env` sets it).
- Send gRPC metadata keys in lowercase (`authorization`). `tritonclient.http` adds entries to the `headers` dict it is given, so pass a fresh dict on every call.
- Expect about 40 ms per ACT call with three 640×480 cameras from the login node.
- After editing `Dockerfile` (`tritonserver:26.08-vllm-python-py3`, vLLM 0.27 and torch 2.14 built for sm_103, plus the ONNX Runtime backend from `tritonserver:26.08-py3`), rebuild with `services/ctl.sh up --build`.
- A vLLM model runs as one instance on one whole GPU, GPU 1 unless the model version's tag `triton.gpu` names another; don't split one across GPUs (the 122B hit uncorrectable NVLink errors). Its engine arguments are `VLLM_ENGINE` in `sync.py` (32k context, 90% of the GPU, up to 4 images per prompt); tag `triton.vllm` with a JSON object to override them for one version.
- vLLM models read their weights in place from the `mlflow` bucket (mounted read-only at `/mlflow`), so nothing is copied into the `triton` bucket. Expect about 4 min to load the 122B (230 GiB of weights, then compilation), so a sync that loads it takes that long.
- Expect about 1.7 s for 17 concurrent requests with one 1024×1024 image and ~90 output tokens each on the 122B (~900 tokens/s together); the first request after a load takes ~20 s.
- `sync.py` is the reconciler. It runs in the same image with the checkout mounted, so a change to it takes effect on the next sync without a rebuild. After changing `backends/` or a packager, dispatch `sync-triton` with `force` set to the affected models to repackage them.
- The server reads the bucket directory read-only and reloads every model on restart. Leave `models/.keep` in place: the gateway deletes an empty folder, and Triton loses its repository with it.
- Metrics are on port 8002 inside the compose network; nothing scrapes them.
