"""Triton Python backend: SAM 2.1 masks from boxes (transformers Sam2Model), on one GPU.

    inputs   image  BYTES [1]      encoded RGB
             boxes  FP32  [-1, 4]  x0, y0, x1, y1 in pixels
    output   masks  BOOL  [-1, H, W]  one mask per box (the single best one), at the image's size
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # _common.py sits next to model.py
from _common import get, image, respond, setup


class TritonPythonModel:
    def initialize(self, args):
        from transformers import Sam2Model, Sam2Processor
        weights, self.device = setup(args)
        self.proc = Sam2Processor.from_pretrained(weights)
        self.model = Sam2Model.from_pretrained(weights).to(self.device).eval()

    def execute(self, requests):
        return respond(self.segment, requests)

    def segment(self, req):
        from PIL import Image
        img = image(get(req, "image").reshape(-1)[0])
        boxes = get(req, "boxes").reshape(-1, 4).astype(float).tolist()
        if not boxes:
            return {"masks": np.zeros((0, *img.shape[:2]), bool)}
        x = self.proc(images=Image.fromarray(img), input_boxes=[boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self.model(**x, multimask_output=False)
        masks = self.proc.post_process_masks(o.pred_masks.cpu(), x["original_sizes"])[0][:, 0].numpy()
        return {"masks": masks.astype(bool)}
