"""X-WAM's scripts/train_sft.py, run unchanged but for two things it hard-codes, set here from outside the submodule:

- the data config: train_sft.py reads configs/data/<dataset>.yaml from the X-WAM tree; with dataset=xwam_behavior it
  reads $XWAM_DATA_CONFIG instead (train_xwam.sbatch writes it: upstream's robodojo.yaml with this dataset's path,
  quantile statistics and video size);
- the DeepSpeed ZeRO stage: train_sft.py builds Lightning's DeepSpeedStrategy with its default stage 2;
  $XWAM_ZERO_STAGE sets it (FastWAM on this data turned a weight NaN under ZeRO-1 and trained under stage 0);
- resuming: train_sft.py only loads weights (pretrained_checkpoint); $XWAM_RESUME_CKPT (a run's checkpoints/<name>.ckpt)
  is passed to Lightning's fit, which restores the weights, optimizer, schedule and step from it.

    cd <X-WAM> && XWAM_DATA_CONFIG=<yaml> XWAM_ZERO_STAGE=0 torchrun ... train_xwam.py dataset=xwam_behavior <overrides>
"""
import functools
import os
import runpy
import sys

from omegaconf import OmegaConf
import lightning
import lightning.pytorch.strategies as strategies

_load = OmegaConf.load


def load(path, *a, **k):
    if str(path) == "configs/data/xwam_behavior.yaml":
        path = os.environ["XWAM_DATA_CONFIG"]
    return _load(path, *a, **k)


OmegaConf.load = load
if os.environ.get("XWAM_ZERO_STAGE"):
    strategies.DeepSpeedStrategy = functools.partial(strategies.DeepSpeedStrategy, stage=int(os.environ["XWAM_ZERO_STAGE"]))
if os.environ.get("XWAM_RESUME_CKPT"):
    lightning.Trainer.fit = functools.partialmethod(lightning.Trainer.fit, ckpt_path=os.environ["XWAM_RESUME_CKPT"])
sys.argv = ["scripts/train_sft.py"] + sys.argv[1:]
runpy.run_path("scripts/train_sft.py", run_name="__main__")
