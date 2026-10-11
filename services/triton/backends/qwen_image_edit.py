"""Triton Python backend: Qwen-Image-Edit-2511 (diffusers QwenImageEditPlusPipeline, bf16, on one GPU).

    inputs   images           BYTES [-1]  1 to 3 encoded RGB images, picture 1 first
             prompt           BYTES [1]
             negative_prompt  BYTES [1]   optional, default " "
             steps            INT32 [1]   optional, default 40
             seed             INT32 [1]   optional, default 0
             true_cfg_scale   FP32  [1]   optional, default 4.0
             lora             BYTES [1]   optional, default "" (none); "angles": the camera-angle LoRA
                                          (fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA, prompts "<sks> <view> <elevation>
                                          <distance>", e.g. "<sks> back-left quarter view elevated shot medium shot")
             lora_scale       FP32  [1]   optional, default 1.0
    output   image            BYTES [1]   PNG at the pipeline's output size

The LoRA (parameter weights_lora_angles: its registered model's folder) loads once as a named adapter and is switched
off between requests that don't ask for it, so those run the base model exactly.
"""

import json
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
        params = json.loads(args["model_config"])["parameters"]
        self.loras = []
        if "weights_lora_angles" in params:
            self.pipe.load_lora_weights(params["weights_lora_angles"]["string_value"], adapter_name="angles",
                                        weight_name="qwen-image-edit-2511-multiple-angles-lora.safetensors")
            self.loras.append("angles")
            self.pipe.disable_lora()

    def execute(self, requests):
        return respond(self.edit, requests)

    def edit(self, req):
        from PIL import Image
        imgs = [Image.fromarray(image(b)) for b in get(req, "images").reshape(-1)]
        if not 1 <= len(imgs) <= 3:
            raise ValueError(f"images: 1 to 3, got {len(imgs)}")
        lora = text(req, "lora")
        if lora and lora not in self.loras:
            raise ValueError(f"lora: one of {self.loras or 'none loaded'}, got {lora!r}")
        if lora:
            self.pipe.enable_lora()
            self.pipe.set_adapters([lora], adapter_weights=[float(scalar(req, "lora_scale", 1.0))])
        try:
            return self._edit(req, imgs)
        finally:
            if lora:
                self.pipe.disable_lora()

    def _edit(self, req, imgs):
        res = self.pipe(image=imgs, prompt=text(req, "prompt"), negative_prompt=text(req, "negative_prompt") or " ",
                        num_inference_steps=int(scalar(req, "steps", 40)), true_cfg_scale=float(scalar(req, "true_cfg_scale", 4.0)),
                        generator=torch.Generator(self.device).manual_seed(int(scalar(req, "seed", 0)))).images[0]
        return {"image": np.array([png(res)], dtype=object)}
