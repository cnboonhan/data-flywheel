"""FastWAM's scripts/train.py, run from its checkout, with one fix for fp32 training under DeepSpeed.

    accelerate launch ... adapter/fastwam/train.py <hydra overrides>     (gb300/train_fastwam.sbatch)

Without bf16, DeepSpeed computes no global gradient norm, so accelerator.clip_grad_norm_ returns None and FastWAM's
trainer fails turning it into a logged tensor; here it reports 0.0 instead (DeepSpeed still clips, gradient_clipping
in ds_zero0.json).
"""
import runpy
import sys

import fastwam.trainer as T

for name in dir(T):
    cls = getattr(T, name)
    if isinstance(cls, type) and hasattr(cls, "_apply_dit_only_train_mode"):
        init = cls.__init__

        def patched(self, *a, _init=init, **k):
            _init(self, *a, **k)
            clip = self.accelerator.clip_grad_norm_

            def clip_or_zero(*x, **y):
                norm = clip(*x, **y)
                return 0.0 if norm is None else norm

            self.accelerator.clip_grad_norm_ = clip_or_zero

        cls.__init__ = patched

sys.argv = ["scripts/train.py"] + sys.argv[1:]
runpy.run_path("scripts/train.py", run_name="__main__")
