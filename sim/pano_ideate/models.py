"""The models pano_ideate uses, each served by Triton at its own endpoint (services/triton/triton_endpoint.py), so the
steps and the viewer run on any machine that reaches them, a laptop included, without a GPU.

    | Role    | Variable          | Default model          | Licence    | Used for                                  |
    | VLM     | VLM_ENDPOINT      | Qwen3.5-122B-A10B      | Apache-2.0 | listing objects, planning actions         |
    | EDIT    | EDIT_ENDPOINT     | Qwen-Image-Edit-2511   | Apache-2.0 | end states, removing and placing objects  |
    | DEPTH   | DEPTH_ENDPOINT    | moge-3-vitg            | MIT        | metric depth                              |
    | DETECT  | DETECT_ENDPOINT   | llmdet_large           | Apache-2.0 | finding objects                           |
    | SEGMENT | SEGMENT_ENDPOINT  | sam2.1-hiera-large     | Apache-2.0 | outlining objects                         |

Each variable is a full URL (https://<host:port>/v2/models/<model>) or a model name on the flywheel Triton (TRITON_URL,
else triton.$SERVICE_HOST:$CADDY_PORT); the token is TRITON_TOKEN (or <ROLE>_TOKEN), the CA AWS_CA_BUNDLE (quickstart.md). The VLM's
chat template comes from its Hugging Face tokenizer (VLM_TOKENIZER; a few MB, no weights, no torch).
"""

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "services" / "triton"))
from triton_endpoint import Endpoint  # noqa: E402


def jpg(image: np.ndarray, quality: int = 95) -> bytes:
    """An RGB image as JPEG bytes."""
    return cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()


class Triton:
    def __init__(self):
        self.vlm_ = Endpoint("VLM", "Qwen3.5-122B-A10B")
        self.edit_ = Endpoint("EDIT", "Qwen-Image-Edit-2511")
        self.depth_ = Endpoint("DEPTH", "moge-3-vitg")
        self.detect_ = Endpoint("DETECT", "llmdet_large")
        self.segment_ = Endpoint("SEGMENT", "sam2.1-hiera-large")
        self.url = self.vlm_.host
        self._tokenizer = None

    def __repr__(self):
        return ", ".join(map(repr, (self.vlm_, self.edit_, self.depth_, self.detect_, self.segment_)))

    def vlm(self, text: str, image: np.ndarray | None = None, max_tokens: int = 1000) -> str:
        """One answer from the VLM (greedy, thinking off), about the RGB image if given."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(os.environ.get("VLM_TOKENIZER", "Qwen/Qwen3.5-122B-A10B"))
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": text}]
        prompt = self._tokenizer.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                     tokenize=False, enable_thinking=False)
        body = {"text_input": prompt, "exclude_input_in_output": True,
                "sampling_parameters": json.dumps({"temperature": 0, "max_tokens": max_tokens})}
        if image is not None:
            body["image"] = [base64.b64encode(jpg(image, 92)).decode()]
        return self.vlm_.generate(body)["text_output"].strip()

    def edit(self, images: list[np.ndarray], prompt: str, steps: int = 40, seed: int = 0, cfg: float = 4.0,
             negative: str = " ", lora: str = "", lora_scale: float = 1.0) -> np.ndarray:
        """Qwen-Image-Edit: picture 1 edited as the prompt says (pictures 2 and 3 for reference); RGB. lora "angles"
        turns on the camera-angle LoRA (services/triton/README.md)."""
        out = self.edit_.infer({"images": [png(i) for i in images], "prompt": [prompt.encode()],
                                "negative_prompt": [negative.encode()], "steps": np.array([steps], np.int32),
                                "seed": np.array([seed], np.int32), "true_cfg_scale": np.array([cfg], np.float32),
                                "lora": [lora.encode()], "lora_scale": np.array([lora_scale], np.float32)},
                               ["image"])["image"]
        raw = np.frombuffer(np.ravel(out)[0], np.uint8)
        return cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    def depth(self, image: np.ndarray, fov_x: float, view_size: int = 1024, merge_width: int = 0):
        """MoGe-3: metric distance per pixel and where it is valid, of a perspective view (fov_x degrees) or, with
        fov_x 0, of an equirectangular panorama (split into 20 views of view_size px, merged at merge_width; 0: its
        own width). A panorama takes about 2 min."""
        out = self.depth_.infer({"image": [jpg(image)], "fov_x": np.array([fov_x], np.float32),
                                 "view_size": np.array([view_size], np.int32),
                                 "merge_width": np.array([merge_width], np.int32)}, ["distance", "mask"])
        return out["distance"].astype(np.float32), out["mask"].astype(bool)

    def detect(self, image: np.ndarray, labels: list[str]):
        """LLMDet: boxes (pixels, x0 y0 x1 y1), scores and names of the labels found, after NMS."""
        out = self.detect_.infer({"image": [jpg(image)], "labels": [n.encode() for n in labels]},
                                 ["boxes", "scores", "names"])
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in np.ravel(out["names"])]
        return out["boxes"].reshape(-1, 4), out["scores"].reshape(-1), names

    def segment(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """SAM 2.1: one mask per box (pixels, x0 y0 x1 y1), image-sized."""
        out = self.segment_.infer({"image": [jpg(image)], "boxes": np.asarray(boxes, np.float32).reshape(-1, 4)},
                                  ["masks"])
        return out["masks"].astype(bool)
