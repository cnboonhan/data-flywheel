"""Helpers shared by the Hugging Face model backends (copied next to each one's model.py by sync.py)."""

import io
import json

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image


def setup(args):
    """(weights dir, torch device) from Triton's args: the model config's parameter `weights`, the instance's GPU."""
    import torch
    cfg = json.loads(args["model_config"])
    weights = cfg["parameters"]["weights"]["string_value"]
    dev = int(args["model_instance_device_id"])
    torch.cuda.set_device(dev)
    # This image's cuDNN is unusable on sm_103 (GB300): Qwen-Image-Edit comes out as noise with it, MoGe's transposed
    # convolutions find no engine. PyTorch's own kernels instead, and faster here; each model has its own process.
    torch.backends.cudnn.enabled = False
    return weights, torch.device(f"cuda:{dev}")


def get(req, name, default=None):
    """An input as a numpy array, or default when the optional input is absent."""
    t = pb_utils.get_input_tensor_by_name(req, name)
    return default if t is None else t.as_numpy()


def scalar(req, name, default):
    v = get(req, name)
    return default if v is None else v.reshape(-1)[0].item()


def text(req, name, default=""):
    v = get(req, name)
    return default if v is None else v.reshape(-1)[0].decode()


def image(b) -> np.ndarray:
    """An encoded image (bytes) as RGB uint8."""
    return np.array(Image.open(io.BytesIO(b)).convert("RGB"))


def png(img) -> bytes:
    buf = io.BytesIO()
    (img if isinstance(img, Image.Image) else Image.fromarray(img)).save(buf, format="PNG")
    return buf.getvalue()


def respond(fn, requests):
    """One response per request; one bad request must not fail the others."""
    out = []
    for req in requests:
        try:
            out.append(pb_utils.InferenceResponse([pb_utils.Tensor(k, v) for k, v in fn(req).items()]))
        except Exception as e:
            out.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(f"{type(e).__name__}: {e}")))
    return out
