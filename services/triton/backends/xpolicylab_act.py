"""Triton Python backend for an XPolicyLab ACT checkpoint (copied in as <model>/<version>/model.py by sync.py).

The version directory holds this file, XPolicyLab's detr/ package, policy_last.ckpt, dataset_stats.pkl and act.json
(the training arguments). Stateless: one request = one observation, the response is the whole action chunk, which the
client executes (no temporal aggregation, so many clients can share an instance).

    inputs   state      FP32  [B, D]        packed robot state, per arm [arm, ee] in the embodiment's arm order
             <camera>   UINT8 [B, H, W, 3]  RGB, any size (resized to 640x480 as XPolicyLab's ACT does)
    output   action     FP32  [B, chunk, D]
"""

import json
import os
import sys
import types

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.nn import functional as F


class TritonPythonModel:
    def initialize(self, args):
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path[:0] = [here, os.path.join(here, "detr")]   # detr.* as a package; its backbone imports util.misc absolutely
        sys.modules.setdefault("IPython", types.SimpleNamespace(embed=lambda *a, **k: None))   # imported, never used
        cfg = json.load(open(os.path.join(here, "act.json")))
        os.environ["ACT_ACTION_DIM"] = str(cfg["action_dim"])   # read by detr's model builder

        import detr.models.backbone
        detr.models.backbone.is_main_process = lambda: False   # no ImageNet download: the checkpoint has the weights
        from detr.act_policy import ACT
        from argparse import Namespace

        dev = int(args["model_instance_device_id"]) if args["model_instance_kind"] == "GPU" else None
        if dev is not None:
            torch.cuda.set_device(dev)
        self.device = torch.device(f"cuda:{dev}" if dev is not None else "cpu")
        cfg.update(ckpt_dir=here, device=str(self.device), temporal_agg=False)
        self.act = ACT(cfg, Namespace(**cfg))
        self.cameras = cfg["camera_names"]

    def execute(self, requests):
        out = []
        for req in requests:
            try:
                state = pb_utils.get_input_tensor_by_name(req, "state").as_numpy()
                imgs = []
                for cam in self.cameras:
                    t = torch.from_numpy(pb_utils.get_input_tensor_by_name(req, cam).as_numpy()).to(self.device)
                    t = t.permute(0, 3, 1, 2).float() / 255.0
                    imgs.append(F.interpolate(t, size=(480, 640), mode="bilinear", align_corners=False))
                image = torch.stack(imgs, dim=1)   # [B, cams, 3, 480, 640]
                qpos = torch.from_numpy(self.act.pre_process(state)).float().to(self.device)
                with torch.no_grad():
                    a = self.act.policy(qpos, image).cpu().numpy()
                action = self.act.post_process(a).astype(np.float32)
                out.append(pb_utils.InferenceResponse([pb_utils.Tensor("action", action)]))
            except Exception as e:   # one bad request must not fail the batch
                out.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(f"{type(e).__name__}: {e}")))
        return out
