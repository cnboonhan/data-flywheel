# triton

Triton Inference Server on two GPUs of a node of its own (`TRITON_NODE`, `TRITON_GPU0`, `TRITON_GPU1` in [`.env.example`](../.env.example); `compose.yml`, started by `ctl.sh up`), serving models from the MLflow registry at `triton.$SERVICE_HOST:$CADDY_PORT` (HTTP and gRPC, header `Authorization: Bearer $TRITON_TOKEN`).

## Serve a model

1. Set the alias `triton` on the registered model version to serve; remove it to unload.
   ```python
   mlflow.MlflowClient().set_registered_model_alias("ACT.arx_x5.RoboDojo.stack_bowls", "triton", "3")
   mlflow.MlflowClient().delete_registered_model_alias("ACT.arx_x5.RoboDojo.stack_bowls", "triton")
   ```
2. Dispatch the Gitea workflow `sync-triton` ([how](../gitea/README.md)). It packages the version into `s3://triton/models/<name>/<version>/`, loads it and runs a test inference.
3. Check the outcome in [Tracking](#tracking).

## Call a model

Use `TRITON_URL` and `TRITON_TOKEN` from `flywheel.env`:

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

Call the image and geometry models (Python backends, below) with `tritonclient.http`: images go as encoded bytes (JPEG or PNG), results come back as binary tensors. From a laptop, open the tunnel and set `TRITON_URL`, `TRITON_TOKEN` and the CA as in the [quickstart](../../quickstart.md).

```python
import os, numpy as np, tritonclient.http as th
c = th.InferenceServerClient(os.environ["TRITON_URL"], ssl=True, ssl_options={"ca_certs": os.environ["AWS_CA_BUNDLE"]},
                             network_timeout=900)

def tensor(name, values, dtype):   # dtype: BYTES, FP32 or INT32
    a = np.asarray(values, dtype={"BYTES": object, "FP32": np.float32, "INT32": np.int32}[dtype])
    t = th.InferInput(name, list(a.shape), dtype)
    t.set_data_from_numpy(a)
    return t

jpg = open("view.jpg", "rb").read()
auth = lambda: {"authorization": "Bearer " + os.environ["TRITON_TOKEN"]}   # a fresh dict per call
r = c.infer("moge-3-vitg", [tensor("image", [jpg], "BYTES"), tensor("fov_x", [90.0], "FP32")], headers=auth())
distance = r.as_numpy("distance")                                          # metres, [H, W]
r = c.infer("Qwen-Image-Edit-2511", [tensor("images", [jpg], "BYTES"), tensor("prompt", [b"Remove the bucket."], "BYTES")],
            headers=auth())
png = r.as_numpy("image")[0]                                               # the edited image, PNG bytes
```

The sim tools ([pano_ideate](../../sim/pano_ideate/README.md), [asset_gen](../../sim/asset_gen/README.md), [colmap_splat](../../sim/colmap_splat/README.md#train)) call models through `triton_endpoint.py`: one `Endpoint(role, default_model)` per model they need, configured by `<ROLE>_ENDPOINT` (a model name on this Triton, or a full `https://<host:port>/v2/models/<model>` on any Triton), `<ROLE>_TOKEN` (else `TRITON_TOKEN`) and `<ROLE>_CA` (else `AWS_CA_BUNDLE`). Each tool's README lists its roles.

```python
from triton_endpoint import Endpoint                  # services/triton on sys.path
depth = Endpoint("DEPTH", "moge-3-vitg")              # DEPTH_ENDPOINT overrides
distance = depth.infer({"image": [jpg], "fov_x": np.array([90.0], np.float32)}, ["distance"])["distance"]
```

| Run param `model` | Backend | Inputs | Output |
|---|---|---|---|
| `ACT` | Python (`backends/xpolicylab_act.py`) | `state` FP32 `[B, D]`, one UINT8 `[B, H, W, 3]` RGB per camera | `action` `[B, chunk, D]` |
| `MLP` | ONNX Runtime | `state` FP32 `[B, D]` | `action` `[B, 1, D]` |
| Hugging Face model (`download-models-hf`) | vLLM | `text_input`, optional `image` (base64), `sampling_parameters` (JSON) | `text_output` |
| `Qwen-Image-Edit-2511` | Python (`backends/qwen_image_edit.py`) | `images` BYTES `[1-3]` (picture 1 first), `prompt` BYTES `[1]`; optional `negative_prompt`, `steps` INT32 (40), `seed` INT32 (0), `true_cfg_scale` FP32 (4.0) | `image` BYTES `[1]` (PNG) |
| `moge-3-vitg` | Python (`backends/moge.py`) | `image` BYTES `[1]`, `fov_x` FP32 `[1]` (degrees; 0 = equirectangular panorama); optional `view_size` INT32 (1024), `merge_width` INT32 (0 = full width) | `distance` FP32 `[H, W]` (metres), `mask` BOOL `[H, W]` |
| `llmdet_large` | Python (`backends/llmdet.py`) | `image` BYTES `[1]`, `labels` BYTES `[N]`; optional `box_threshold` (0.35), `text_threshold` (0.25), `nms` (0.7) FP32 | `boxes` FP32 `[K, 4]` (pixels), `scores` FP32 `[K]`, `names` BYTES `[K]` |
| `sam2.1-hiera-large` | Python (`backends/sam2.py`) | `image` BYTES `[1]`, `boxes` FP32 `[K, 4]` (pixels) | `masks` BOOL `[K, H, W]` |
| `TripoSG` | Python (`backends/triposg.py`) | `image` BYTES `[1]` (RGBA PNG cutout of one object); optional `seed` INT32 (0), `steps` INT32 (50), `guidance` FP32 (7.0), `faces` INT32 (-1 = all; else simplify to about that many) | `mesh` BYTES `[1]` (GLB, shape only: y up, centred, within ±1) |

A Hugging Face model goes to the Python backend that `HF_BACKENDS` in `sync.py` names for its repo (or its version's tag `triton.backend`), otherwise to vLLM. Other models are tagged `unsupported`; to add one, write a `package_<model>` function in `sync.py`, or a backend in `backends/` with its inputs and outputs in `PYTHON_IO`.

## Tracking

| Where | What |
|---|---|
| MLflow version tags | `triton.status` (`ready`, `failed`, `unsupported`, `unloaded`), `triton.error`, `triton.synced_at` |
| `GET /v2/models/<name>/config` → `parameters` | MLflow model, version, run, embodiment, mix, data fingerprint, git commit |
| `s3://triton/sync/status.json`, `sync/log/<time>.json` | current state; one file per change |
| `sync.py --status`, `sync-triton` run log | Triton's loaded models with their MLflow source |

## Notes

- If `TRITON_NODE` is also a Slurm node, jobs placed there may share its GPUs with Triton; keep GPU-heavy jobs off it (`--exclude=$TRITON_NODE`) until its GPUs are removed from Slurm.
- Send gRPC metadata keys in lowercase (`authorization`). `tritonclient.http` adds entries to the `headers` dict it is given, so pass a fresh dict on every call.
- Expect about 40 ms per ACT call with three 640×480 cameras from the login node.
- After editing `Dockerfile` (`tritonserver:26.08-vllm-python-py3`, vLLM 0.27 and torch 2.14 built for sm_103, plus the ONNX Runtime backend from `tritonserver:26.08-py3`), rebuild on the node with `services/ctl.sh triton build`, then `services/ctl.sh triton up -d` (and in the stack for the sync-triton job: `services/ctl.sh build`).
- A vLLM model runs as one instance on one whole GPU, the second of Triton's GPUs (`TRITON_GPU1`, index 1 inside it) unless the model version's tag `triton.gpu` names another index; don't split one across GPUs (the 122B hit uncorrectable NVLink errors). Its engine arguments are `VLLM_ENGINE` in `sync.py` (32k context, 90% of the GPU, up to 4 images per prompt); tag `triton.vllm` with a JSON object to override them for one version. A resync of a vLLM model unloads it before loading the new copy (Triton otherwise loads the new copy first, and two don't fit its GPU share), so it is down for its load time. vLLM models report their tokens on the metrics port (`vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, ...; Grafana **Flywheel → Model usage**).
- The Python-backend Hugging Face models run as one instance each on the first of Triton's GPUs (`TRITON_GPU0`, index 0 inside it; `TRITON_PYTHON_GPU`, or tag `triton.gpu`), about 71 GB together (the image editor 68 GB). They run with cuDNN off: in this image it makes Qwen-Image-Edit return noise and has no kernel for MoGe's transposed convolutions on GB300.
- Expect on the Python backends: an image edit at 1024×1024 in ~16 s (40 steps), MoGe on one 1024 px view in ~0.6 s and on a 4096×2048 panorama in ~110 s (its 20 views, then the merge on the CPU), LLMDet and SAM in well under a second per view.
- TripoSG ([VAST-AI/TripoSG](https://huggingface.co/VAST-AI/TripoSG), MIT) returns a shape without colours. Its turn about the vertical is its own, close to the input's view from +z (image right +x) but not exactly, so fit the yaw against the input. Give it a cutout: it skips TripoSG's RMBG-1.4 background removal (non-commercial). Expect ~12 s per mesh at 50 steps (~1.6M faces before `faces` simplifies it).
- vLLM models read their weights in place from the `mlflow` bucket (mounted read-only at `/mlflow`), so nothing is copied into the `triton` bucket. Expect about 4 min to load the 122B (230 GiB of weights, then compilation), so a sync that loads it takes that long.
- Expect about 1.7 s for 17 concurrent requests with one 1024×1024 image and ~90 output tokens each on the 122B (~900 tokens/s together); the first request after a load takes ~20 s.
- `sync.py` is the reconciler. It runs in the same image with the checkout mounted, so a change to it takes effect on the next sync without a rebuild. After changing `backends/` or a packager, dispatch `sync-triton` with `force` set to the affected models to repackage them.
- The server reads the bucket directory read-only and reloads every model on restart. Leave `models/.keep` in place: the gateway deletes an empty folder, and Triton loses its repository with it.
- Triton's ports are published on `TRITON_NODE` (8100 HTTP, 8101 gRPC, 8102 metrics), so every API but health requires the header `triton-key: $TRITON_KEY`; Caddy adds it for `triton.$SERVICE_HOST` and `sync-triton` sends it. [Prometheus](../prometheus/README.md) collects the metrics (job `triton`). `ctl.sh triton logs -f` shows its log.
