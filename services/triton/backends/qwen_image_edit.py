"""Triton Python backend: Qwen-Image-Edit-2511 (diffusers QwenImageEditPlusPipeline, bf16, on one GPU).

    inputs   images           BYTES [-1]  1 to 3 encoded RGB images, picture 1 first
             prompt           BYTES [1]
             negative_prompt  BYTES [1]   optional, default " "
             steps            INT32 [1]   optional, default 40
             seed             INT32 [1]   optional, default 0
             true_cfg_scale   FP32  [1]   optional, default 4.0
    output   image            BYTES [1]   PNG at the pipeline's output size
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # _common.py sits next to model.py
from _common import get, image, png, respond, scalar, setup, text


class TritonPythonModel:
    def initialize(self, args):
        from diffusers import QwenImageEditPlusPipeline
        weights, self.device = setup(args)
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(weights, torch_dtype=torch.bfloat16).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def execute(self, requests):
        return respond(self.edit, requests)

    def edit(self, req):
        from PIL import Image
        imgs = [Image.fromarray(image(b)) for b in get(req, "images").reshape(-1)]
        if not 1 <= len(imgs) <= 3:
            raise ValueError(f"images: 1 to 3, got {len(imgs)}")
        # Defaults as in the model card's example: 40 steps, true_cfg_scale 4.0, negative prompt " ", seed 0.
        res = self.pipe(image=imgs, prompt=text(req, "prompt"), negative_prompt=text(req, "negative_prompt") or " ",
                        num_inference_steps=int(scalar(req, "steps", 40)), true_cfg_scale=float(scalar(req, "true_cfg_scale", 4.0)),
                        generator=torch.Generator(self.device).manual_seed(int(scalar(req, "seed", 0)))).images[0]
        return {"image": np.array([png(res)], dtype=object)}
