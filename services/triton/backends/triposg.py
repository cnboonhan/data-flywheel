"""Triton Python backend: TripoSG, one image of an object to its shape (a mesh, no colours), on one GPU.

    inputs   image     BYTES [1]  RGBA PNG cutout of the object (transparent around it)
             seed      INT32 [1]  optional, default 0
             steps     INT32 [1]  optional, sampling steps (default 50)
             guidance  FP32  [1]  optional, classifier-free guidance scale (default 7.0)
             faces     INT32 [1]  optional, simplify to about this many faces (default -1: keep them all)
    outputs  mesh      BYTES [1]  the mesh as GLB: y up, centred, within [-1, 1]; its turn about y is TripoSG's own,
                                  roughly the input's view from +z (image right +x) but not exactly (a three-quarter
                                  view can come out side-on), so fit the yaw against the input

The cutout is prepared as TripoSG's own prepare_image does after background removal (on white, cropped to the
object, padded 10% and squared), without its RMBG-1.4 net (non-commercial): the caller cuts the object out. The
surface is extracted on TripoSG's flash decoder grid with scikit-image's marching cubes in place of diso's DiffDMC
(a CUDA extension).
"""

import io
import os
import sys
import types

import numpy as np
import torch
import trimesh
from PIL import Image


class _MarchingCubes:
    """diso.DiffDMC's call as TripoSG's flash decoder makes it: vertices (grid units) and faces of an SDF's zero level,
    where the grid's unevaluated (NaN) cells are skipped."""

    def __init__(self, dtype=None):
        pass

    def to(self, device):
        self.device = device
        return self

    def __call__(self, sdf, deform=None, return_quads=False, normalize=False):
        from skimage.measure import marching_cubes
        v = sdf.float().cpu().numpy()
        ok = np.isfinite(v)
        cell = ok.copy()                                  # a cell counts when all 8 of its corners were evaluated
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    cell[:-1, :-1, :-1] &= ok[dx:dx + v.shape[0] - 1, dy:dy + v.shape[1] - 1, dz:dz + v.shape[2] - 1]
        cell[-1], cell[:, -1], cell[:, :, -1] = False, False, False
        verts, faces, _, _ = marching_cubes(np.where(ok, v, 1.0), 0.0, mask=cell)
        # TripoSG reverses DiffDMC's faces; these come out the other way round, so pre-reverse them.
        return torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces[:, ::-1].copy())


sys.modules["diso"] = types.SimpleNamespace(DiffDMC=_MarchingCubes)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # _common.py sits next to model.py
from _common import get, respond, scalar, setup  # noqa: E402


def prepare(png: bytes) -> Image.Image:
    """The cutout on white, cropped to its alpha's bounds, padded by 10% of the longer side and squared."""
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA")).astype(np.float32) / 255
    a = img[..., 3:]
    ys, xs = np.nonzero(a[..., 0] > 0)
    if not len(xs):
        raise ValueError("image: no object (alpha is 0 everywhere)")
    rgb = (img[..., :3] * a + (1 - a))[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = rgb.shape[:2]
    pad = int(max(h, w) * 0.1)
    side = max(h, w) + 2 * pad
    out = np.ones((side, side, 3), np.float32)
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = rgb
    return Image.fromarray((out * 255).astype(np.uint8))


class TritonPythonModel:
    def initialize(self, args):
        weights, self.device = setup(args)
        from triposg.pipelines.pipeline_triposg import TripoSGPipeline
        # float16, steps 50, guidance 7.0 and faces -1 (run) are the defaults of TripoSG's scripts/inference_triposg.py.
        self.pipe = TripoSGPipeline.from_pretrained(weights).to(self.device, torch.float16)
        self.pipe.set_progress_bar_config(disable=True)

    def execute(self, requests):
        return respond(self.run, requests)

    def run(self, req):
        image = prepare(get(req, "image").reshape(-1)[0])
        seed, faces = int(scalar(req, "seed", 0)), int(scalar(req, "faces", -1))   # seed: any fixed one (upstream 42)
        with torch.no_grad():
            v, f = self.pipe(image=image, generator=torch.Generator(self.device).manual_seed(seed),
                             num_inference_steps=int(scalar(req, "steps", 50)),
                             guidance_scale=float(scalar(req, "guidance", 7.0))).samples[0]
        if v is None:
            raise RuntimeError("no surface found")
        mesh = trimesh.Trimesh(v.astype(np.float32), np.ascontiguousarray(f))
        mesh = max(mesh.split(only_watertight=False), key=lambda m: len(m.faces))     # drop floating bits
        if 0 < faces < len(mesh.faces):
            import fast_simplification
            v2, f2 = fast_simplification.simplify(mesh.vertices, mesh.faces, target_count=faces)
            mesh = trimesh.Trimesh(v2, f2)
        if mesh.volume < 0:
            mesh.invert()
        return {"mesh": np.array([mesh.export(file_type="glb")], dtype=object)}
