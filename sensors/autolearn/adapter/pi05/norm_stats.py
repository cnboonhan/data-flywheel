"""openpi's norm stats (scripts/compute_norm_stats.py) for an R1 Pro config, from the parquet alone.

    venv-pi05/bin/python norm_stats.py pi05_r1pro_picking_up_cup

openpi's script runs the whole dataset through the loader, decoding three videos per frame only to drop the images;
here the same state and action chunks are built from the parquet (action_horizon frames, clamped at the episode's end
as LeRobot's delta_timestamps are), put through the config's own data transforms (the delta actions), and fed to the
same RunningStats, written where training reads them: <assets_base_dir>/<config>/<repo_id>.

One change to openpi's stats: a dimension whose 1-99% quantile span is under MIN_SPAN gets its full min..max range
instead. The R1 Pro's torso joints 2-4 barely move in these demonstrations but spike now and then; their quantiles span
~0.001, which put the spikes at up to 500 after normalization, and the save test's loss climbed from 0.2 to 36 in 15 steps.
"""
import sys

import numpy as np
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as _transforms
import pyarrow.parquet as pq
from lerobot.utils.constants import HF_LEROBOT_HOME

import r1pro

r1pro.register()
cfg = _config.get_config(sys.argv[1])
dc = cfg.data.create(cfg.assets_dirs, cfg.model)
h = cfg.model.action_horizon
t = pq.read_table(HF_LEROBOT_HOME / dc.repo_id / "data", columns=["observation.state", "action", "episode_index"])
state = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
action = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
ep = t["episode_index"].to_numpy()
last = np.searchsorted(ep, ep, side="right") - 1         # each frame's episode end (rows are in episode order)
idx = np.minimum(np.arange(len(ep))[:, None] + np.arange(h)[None], last[:, None])
delta = [t_ for t_ in dc.data_transforms.inputs if isinstance(t_, _transforms.DeltaActions)]
MIN_SPAN = 0.01      # rad (0.6 deg), or gripper fraction: well below any arm joint's span here (0.4 rad and up)
stats = {k: normalize.RunningStats() for k in ("state", "actions")}
lo = {k: np.inf for k in stats}
hi = {k: -np.inf for k in stats}
for s in range(0, len(ep), 4096):
    b = {"state": state[s:s + 4096], "actions": action[idx[s:s + 4096]]}
    for d in delta:
        b = d(b)
    for k in stats:
        stats[k].update(b[k])
        flat = b[k].reshape(-1, b[k].shape[-1])
        lo[k], hi[k] = np.minimum(lo[k], flat.min(0)), np.maximum(hi[k], flat.max(0))
norm = {}
for k, v in stats.items():
    n = v.get_statistics()
    narrow = n.q99 - n.q01 < MIN_SPAN
    norm[k] = normalize.NormStats(mean=n.mean, std=n.std, q01=np.where(narrow, lo[k], n.q01), q99=np.where(narrow, hi[k], n.q99))
    print(k, "full range for dims", np.flatnonzero(narrow).tolist())
out = cfg.assets_dirs / dc.repo_id
normalize.save(out, norm)
print(f"{len(ep)} frames -> {out}")
