"""Triton Python backend: MoGe metric geometry (MoGe-3 or MoGe-2, by the weights' repo), on one GPU.

    inputs   image        BYTES [1]   encoded RGB
             fov_x        FP32  [1]   horizontal field of view in degrees; 0: the image is a full equirectangular panorama
             view_size    INT32 [1]   optional, panorama: px of each of MoGe's 20 perspective views (default 1024)
             merge_width  INT32 [1]   optional, panorama: width the views are merged at (default 0: the panorama's)
    outputs  distance     FP32  [H, W]  metric distance from the camera per pixel (the point map's norm)
             mask         BOOL  [H, W]  where it is valid

A panorama runs MoGe on its 20-view split, merges the views' depths (MoGe's gradient-domain merge, which loses the
absolute scale) and rescales the result to the views' metric distances, as sim/pano_splat does.
"""

import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # _common.py sits next to model.py
from _common import get, image, respond, scalar, setup


class TritonPythonModel:
    def initialize(self, args):
        weights, self.device = setup(args)
        repo = json.loads(args["model_config"])["parameters"].get("hf_repo", {}).get("string_value", "")
        if "moge-2" in repo:
            from moge.model.v2 import MoGeModel
        else:
            from moge.model.v3 import MoGeModel
        self.model = MoGeModel.from_pretrained(os.path.join(weights, "model.pt")).to(self.device).eval()

    def execute(self, requests):
        return respond(self.run, requests)

    def infer(self, views, fov):
        x = torch.tensor(np.stack(views) / 255, dtype=torch.float32, device=self.device).permute(0, 3, 1, 2)
        with torch.no_grad():
            o = self.model.infer(x, fov_x=torch.tensor(fov, dtype=torch.float32, device=self.device), apply_mask=False)
        return o["points"].norm(dim=-1).float().cpu().numpy(), o["mask"].cpu().numpy()

    def run(self, req):
        img = image(get(req, "image").reshape(-1)[0])
        fov = float(scalar(req, "fov_x", 0.0))
        if fov > 0:
            d, m = self.infer([img], [fov])
            return {"distance": d[0].astype(np.float32), "mask": m[0].astype(bool)}
        d, m = self.panorama(img, int(scalar(req, "view_size", 1024)), int(scalar(req, "merge_width", 0)))
        return {"distance": d.astype(np.float32), "mask": m}

    def panorama(self, image, view_size, merge_width):
        import utils3d_moge as utils3d
        from moge.utils.panorama import (get_panorama_cameras, merge_panorama_depth, spherical_uv_to_directions,
                                         split_panorama_image)
        h, w = image.shape[:2]
        extr, intr = get_panorama_cameras()
        views = split_panorama_image(image, extr, intr, view_size)
        dist, masks = [], []
        for i in range(0, len(views), 4):
            fov = np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(intr[i:i + 4])))[0]
            d, m = self.infer(views[i:i + 4], fov)
            dist += list(d)
            masks += list(m)
        mw = min(merge_width or w, w)
        mh = mw * h // w
        merged, valid = merge_panorama_depth(mw, mh, dist, masks, extr, intr)
        dirs = spherical_uv_to_directions(utils3d.np.uv_map(mh, mw))
        ratios = []
        for d, m, e, k in zip(dist, masks, extr, intr):
            uv, z = utils3d.np.project_cv(dirs, extrinsics=e, intrinsics=k)
            ok = (z > 0) & (uv > 0).all(-1) & (uv < 1).all(-1) & valid
            px = utils3d.np.uv_to_pixel(uv[ok], d.shape).astype(int)
            ok_px = m[px[:, 1], px[:, 0]]
            ratios.append(d[px[ok_px, 1], px[ok_px, 0]] / merged[ok][ok_px])
        merged *= float(np.median(np.concatenate(ratios)))
        return (cv2.resize(merged, (w, h), interpolation=cv2.INTER_LINEAR),
                cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
