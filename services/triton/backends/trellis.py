"""Triton Python backend: TRELLIS (image-large) image to 3D, the mesh with per-vertex colours only, on one GPU.

    inputs   images    BYTES [N]  RGBA PNG cutouts of one object (transparent around it), one or more viewpoints
             seed      INT32 [1]  optional, default 0
             steps     INT32 [1]  optional, sampling steps of both stages (default 12)
             simplify  FP32  [1]  optional, fraction of the faces to remove (default 0.9; 0 keeps them all)
    outputs  glb       BYTES [1]  the mesh as GLB, vertex colours (COLOR_0), y up as TRELLIS exports it

Only the mesh decoder runs: TRELLIS's textured-GLB export renders through nvdiffrast and its Gaussians through
diff-gaussian-rasterization, both licensed for non-commercial use only, so neither is installed and the colours are
the mesh decoder's per-vertex ones. Modules TRELLIS imports but this path never calls are stubbed: rembg (the inputs
are cutouts), open3d (the text-to-3D pipeline), utils3d and plyfile (Gaussians only), kaolin (FlexiCubes uses one tensor check from it). The image
encoder, DINOv2 with registers, loads from its own weights (parameter weights_dinov2) through transformers instead
of torch.hub's download. Sparse convolutions run on torchsparse (its weights converted from spconv's layout as they
load), attention on xformers.
"""

import io
import os
import sys
import types

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPARSE_BACKEND", "torchsparse")


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


_stub("rembg")
_stub("open3d", geometry=types.SimpleNamespace(TriangleMesh=object))   # in a type annotation
_stub("utils3d")
_stub("plyfile", PlyData=None, PlyElement=None)
_stub("kaolin")
_stub("kaolin.utils")
_stub("kaolin.utils.testing", check_tensor=lambda *a, **k: True)

import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402

import xformers.ops  # noqa: E402

# xformers picks FlashAttention-3 (Hopper) by default, which this image has no kernels for on sm_103 (GB300): its
# FlashAttention-2 op instead (fp16, bf16), cutlass otherwise.
_attention = xformers.ops.memory_efficient_attention


def _attention_sm103(q, *a, op=None, **k):
    half = q.dtype in (torch.float16, torch.bfloat16)
    return _attention(q, *a, op=op or (xformers.ops.MemoryEfficientAttentionFlashAttentionOp if half
                                       else xformers.ops.MemoryEfficientAttentionCutlassOp), **k)


xformers.ops.memory_efficient_attention = _attention_sm103

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # _common.py sits next to model.py
from _common import get, respond, scalar, setup  # noqa: E402


def torchsparse_weights(state: dict) -> dict:
    """TRELLIS's checkpoints hold spconv's sparse convolutions (weight: out, kx, ky, kz, in); torchsparse's take a
    kernel per offset (kernel: k^3, in, out, x fastest; in, out for 1x1x1). Dense conv3d weights (out, in, k, k, k)
    pass through."""
    out = {}
    for k, v in state.items():
        s = v.shape
        if k.endswith("conv.weight") and v.ndim == 5 and s[1] == s[2] == s[3] and s[1] in (1, 3):
            kernel = v.permute(3, 2, 1, 4, 0).reshape(s[1] ** 3, s[4], s[0])
            out[k[:-len("weight")] + "kernel"] = kernel if s[1] > 1 else kernel[0]
        else:
            out[k] = v
    return out


class Dinov2(torch.nn.Module):
    """DINOv2 with registers from transformers, answering like torch.hub's: x_prenorm is every token (class,
    registers, patches) after the last block, before the final norm."""

    def __init__(self, path):
        super().__init__()
        from transformers import Dinov2WithRegistersModel
        self.model = Dinov2WithRegistersModel.from_pretrained(path).eval()

    def forward(self, x, is_training=True):
        return {"x_prenorm": self.model(pixel_values=x, output_hidden_states=True).hidden_states[-1]}


class TritonPythonModel:
    def initialize(self, args):
        weights, self.device = setup(args)
        params = json.loads(args["model_config"])["parameters"]
        if "weights_dinov2" not in params:
            raise RuntimeError("register facebook/dinov2-with-registers-large as dinov2-with-registers-large, then resync")
        dinov2 = params["weights_dinov2"]["string_value"]
        import safetensors.torch
        from torchvision import transforms
        from trellis.pipelines import TrellisImageTo3DPipeline
        load_file = safetensors.torch.load_file                   # TRELLIS's model loader imports it per call
        safetensors.torch.load_file = lambda *a, **k: torchsparse_weights(load_file(*a, **k))

        def image_cond_model(pipeline, name):
            if name is None:
                return
            pipeline.models["image_cond_model"] = Dinov2(dinov2)
            pipeline.image_cond_model_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])
        TrellisImageTo3DPipeline._init_image_cond_model = image_cond_model
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(weights)
        for k in ("slat_decoder_gs", "slat_decoder_rf"):      # never used: free their memory
            self.pipeline.models.pop(k, None)
        self.pipeline.to(self.device)

    def execute(self, requests):
        return respond(self.run, requests)

    def run(self, req):
        images = [Image.open(io.BytesIO(b)).convert("RGBA") for b in get(req, "images").reshape(-1)]
        for i in images:            # an opaque image would go to rembg (not installed): take all of it as the object
            if i.getextrema()[3][0] == 255:
                i.putalpha(254)
        seed, steps = int(scalar(req, "seed", 0)), int(scalar(req, "steps", 12))
        simplify = float(scalar(req, "simplify", 0.9))
        sampling = {"steps": steps}
        with torch.no_grad():
            if len(images) > 1:
                out = self.pipeline.run_multi_image(images, seed=seed, formats=["mesh"],
                                                    sparse_structure_sampler_params=sampling,
                                                    slat_sampler_params=sampling)
            else:
                out = self.pipeline.run(images[0], seed=seed, formats=["mesh"],
                                        sparse_structure_sampler_params=sampling, slat_sampler_params=sampling)
        m = out["mesh"][0]
        vertices = m.vertices.float().cpu().numpy()
        faces = m.faces.cpu().numpy()
        colours = m.vertex_attrs[:, :3].float().clamp(0, 1).cpu().numpy()
        if 0 < simplify < 1:
            import fast_simplification
            from scipy.spatial import cKDTree
            v2, faces = fast_simplification.simplify(vertices, faces, target_reduction=simplify)
            colours = colours[cKDTree(vertices).query(v2)[1]]          # each kept vertex: its nearest original's
            vertices = v2
        vertices = vertices @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])     # TRELLIS is z up; glTF y up
        mesh = trimesh.Trimesh(vertices, faces, vertex_colors=(colours * 255).astype(np.uint8), process=False)
        return {"glb": np.array([mesh.export(file_type="glb")], dtype=object)}
