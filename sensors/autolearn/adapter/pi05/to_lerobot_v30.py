"""A lerobot_to_fastwam.py dataset (LeRobot v2.1: 20-value joint state and action, cam_high / cam_left_wrist /
cam_right_wrist) converted to LeRobot v3.0, the only version the lerobot 0.4.4 in openpi's lock reads.

    venv-pi05/bin/python to_lerobot_v30.py <v2.1 dataset> --out $STATE/pi05/lerobot/r1pro_picking_up_cup

The source is mirrored as a tree of symlinks and converted with lerobot's own v2.1 -> v3.0 converter (stream copy, no
re-encoding), so it is left as it is. The output's parent is openpi's HF_LEROBOT_HOME, its name the config's repo_id.
"""
import argparse
import os
import shutil
from pathlib import Path

from lerobot.datasets.v30.convert_dataset_v21_to_v30 import convert_dataset

p = argparse.ArgumentParser()
p.add_argument("src", type=Path)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
src, out = args.src.resolve(), args.out
for d in (out, out.with_name(out.name + "_old"), out.with_name(out.name + "_v30")):
    if d.exists():
        shutil.rmtree(d)
for dirpath, _, files in os.walk(src, followlinks=True):
    rel = Path(dirpath).relative_to(src)
    (out / rel).mkdir(parents=True, exist_ok=True)
    for f in files:
        os.symlink(os.path.realpath(Path(dirpath) / f), out / rel / f)
# the converter edits only its copies: the symlinked source ends up under <out>_old, the v3.0 dataset at <out>
convert_dataset(repo_id=out.name, root=out.parent, push_to_hub=False, force_conversion=True)
shutil.rmtree(out.with_name(out.name + "_old"))
print(f"LeRobot v3.0 dataset at {out}")
