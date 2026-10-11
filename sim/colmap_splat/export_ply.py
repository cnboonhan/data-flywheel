"""Write a 3DGRUT run's splat as PLY (export_last.ply), from its checkpoint, without retraining. Runs in 3DGRUT's venv on
a GPU node (loading the model builds its CUDA tracer). train.py exports PLY itself; this is for runs made before it did.

    sim/colmap_splat/3dgrut/.venv/bin/python sim/colmap_splat/export_ply.py <run dir> [out.ply]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "3dgrut"))
from threedgrut.export import PLYExporter  # noqa: E402
from threedgrut.model.model import MixtureOfGaussians  # noqa: E402

run = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else run / "export_last.ply"
ckpt = torch.load(run / "ckpt_last.pt", weights_only=False)
model = MixtureOfGaussians(ckpt["config"])
model.init_from_checkpoint(ckpt, setup_optimizer=False)
PLYExporter().export(model, out, conf=ckpt["config"])
print(f"{out} ({out.stat().st_size // 2**20} MB, {model.num_gaussians} gaussians)")
