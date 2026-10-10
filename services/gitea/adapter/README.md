# adapter

Converts raw datasets into XPolicyLab's xspark format at `s3://processed/xpolicylab/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`, the input of [train.sbatch](../../slurm/README.md).

| Workflow | Source → bench |
|---|---|
| [`galaxeaOpenWorldDataset_to_xpolicylab`](galaxeaOpenWorldDataset/) | `raw/open_datasets/galaxea-open-world-r1lite/lerobot/<task>.tar.gz` → `galaxeaOpenWorldDataset` |
| [`robodojo_to_xpolicylab`](robodojo/) | `raw/open_datasets/robodojo/<task>/<env_cfg>/data/` (copy) → `RoboDojo` |

1. Run one as described in [gitea/](../README.md#run-a-workflow), with inputs `tasks` (glob, default `*`), `limit` (episodes per task, 0 = all) and `env_cfg` (default `arx_x5`). Start with a small `limit` to check the output.
2. To add a source, create `<repo>/<repo>_to_xpolicylab.yml` (and its script) writing the same layout and a per-task `manifest.json`, then check an episode loads with `XPolicyLab.utils.data_loader.load`.

**Notes**
- Rerun freely: the per-task `manifest.json` maps source to output episodes, so a rerun only adds what's new and skips an unchanged archive without extracting it.
- Galaxea LeRobot v2.1 → xspark: 6 arm joints + 1 gripper per arm (14-D, XPolicyLab's `arx_x5` layout; `env_cfg=arx_x5` is a stand-in label for r1lite until a fork adds it), EE poses reordered to `[x y z qw qx qy qz]`, the three cameras at 480×640 as JPEGs stamped with XPolicyLab's `XPL-RGB1` marker.
- Expect a full Galaxea task to be on the order of 100 episodes and ~130 MB (`Arrange_Fruits_20250819_011`: 114 episodes, 110 064 frames).
