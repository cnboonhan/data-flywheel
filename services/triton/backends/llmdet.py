"""Triton Python backend: LLMDet open-vocabulary detection (transformers zero-shot object detection), on one GPU.

    inputs   image           BYTES [1]   encoded RGB
             labels          BYTES [-1]  names to look for
             box_threshold   FP32  [1]   optional, default 0.35
             text_threshold  FP32  [1]   optional, default 0.25
             nms             FP32  [1]   optional, IoU above which overlapping boxes are dropped (default 0.7)
    outputs  boxes           FP32  [-1, 4]  x0, y0, x1, y1 in pixels
             scores          FP32  [-1]
             names           BYTES [-1]  each box's name, one of labels

The text prompt holds about 256 tokens, so labels go 20 at a time; each answer is mapped back to an input name
(exactly, else by substring), then non-maximum suppression runs over all of them.
"""

import os
import sys

import numpy as np
import torch
from torchvision.ops import nms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # _common.py sits next to model.py
from _common import get, image, respond, scalar, setup


class TritonPythonModel:
    def initialize(self, args):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        weights, self.device = setup(args)
        self.proc = AutoProcessor.from_pretrained(weights)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(weights).to(self.device).eval()

    def execute(self, requests):
        return respond(self.detect, requests)

    def detect(self, req):
        from PIL import Image
        img = image(get(req, "image").reshape(-1)[0])
        labels = [b.decode() for b in get(req, "labels").reshape(-1)]
        bt, tt, iou = (float(scalar(req, k, d)) for k, d in (("box_threshold", 0.35), ("text_threshold", 0.25), ("nms", 0.7)))
        boxes, scores, names = [], [], []
        for j in range(0, len(labels), 20):
            chunk = labels[j:j + 20]
            x = self.proc(images=Image.fromarray(img), text=". ".join(chunk) + ".", return_tensors="pt").to(self.device)
            with torch.no_grad():
                o = self.model(**x)
            r = self.proc.post_process_grounded_object_detection(o, x.input_ids, threshold=bt, text_threshold=tt,
                                                                 target_sizes=[img.shape[:2]])[0]
            for b, s, t in zip(r["boxes"], r["scores"], r["text_labels"]):
                n = next((n for n in chunk if n == t), None) or next((n for n in chunk if t and (t in n or n in t)), None)
                if n:
                    boxes.append(b.float().cpu()), scores.append(s.float().cpu()), names.append(n)
        if not boxes:
            return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
                    "names": np.zeros(0, dtype=object)}
        B, S = torch.stack(boxes), torch.stack(scores)
        keep = nms(B, S, iou).tolist()
        return {"boxes": B[keep].numpy().astype(np.float32), "scores": S[keep].numpy().astype(np.float32),
                "names": np.array([names[k].encode() for k in keep], dtype=object)}
