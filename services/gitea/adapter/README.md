# adapter

Converts raw datasets into XPolicyLab's xspark format at `s3://processed/xpolicylab/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`, the input of [train.sbatch](../../slurm/README.md).

| Workflow | Source → bench |
|---|---|
| [`galaxeaOpenWorldDataset_to_xpolicylab`](galaxeaOpenWorldDataset/) | `raw/open_datasets/galaxea-open-world-r1lite/lerobot/<task>.tar.gz` → `galaxeaOpenWorldDataset` |
| [`robodojo_to_xpolicylab`](robodojo/) | `raw/open_datasets/robodojo/<task>/<env_cfg>/data/` (copy) → `RoboDojo` |

1. Run one ([gitea/](../README.md), step 1) with inputs `tasks` (glob, default `*`), `limit` (episodes per task, 0 = all) and `env_cfg` (default `arx_x5`).
2. Add a source: create `<repo>/<repo>_to_xpolicylab.yml` (+ script) that writes the same layout and a per-task `manifest.json`.

**Notes**
- Incremental: the per-task `manifest.json` maps source → output episodes, so re-runs only add what's new; an unchanged archive is skipped without extracting.
- Galaxea LeRobot v2.1 → xspark: 6 arm joints + 1 gripper per arm (14-D, XPolicyLab's `arx_x5` layout; `env_cfg=arx_x5` is a stand-in label for r1lite until a fork adds it), EE poses reordered to `[x y z qw qx qy qz]`, the three cameras at 480×640 as JPEGs stamped with XPolicyLab's `XPL-RGB1` marker. Verified against `XPolicyLab.utils.data_loader.load`.
- Result: `Arrange_Fruits_20250819_011` in full is 114 episodes, ~130 MB (110 064 frames); `limit = 2` then `limit = 3` added 2 then 1 episode.
