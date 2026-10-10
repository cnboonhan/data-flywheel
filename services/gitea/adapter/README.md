# adapter

Open datasets → XPolicyLab xspark at `s3://processed/xpolicylab/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`. Incremental: a per-task `manifest.json` maps source → output episodes, so re-runs only add what's new.

| Folder | Source → bench |
|---|---|
| [`galaxeaOpenWorldDataset/`](galaxeaOpenWorldDataset/) | `raw/open_datasets/galaxea-open-world-r1lite/lerobot/<task>.tar.gz` → `galaxeaOpenWorldDataset` |
| [`robodojo/`](robodojo/) | `raw/open_datasets/robodojo/<task>/<env_cfg>/data/` (copy) → `RoboDojo` |

Inputs: `tasks` (glob, default `*`), `limit` (episodes per task, 0 = all), `env_cfg` (default `arx_x5`).

Galaxea LeRobot v2.1 → xspark: 6 arm joints + 1 gripper per arm (14-D, XPolicyLab's `arx_x5` layout; `env_cfg=arx_x5` is a stand-in label for r1lite until a fork adds it), EE poses reordered to `[x y z qw qx qy qz]`, the three cameras at 480×640 as JPEGs stamped with XPolicyLab's `XPL-RGB1` marker. Verified against `XPolicyLab.utils.data_loader.load`. `Arrange_Fruits_20250819_011` in full: 114 episodes, ~130 MB (110 064 frames); `limit = 2` then `limit = 3` added 2 then 1 episode; an unchanged archive is skipped without extracting.

New source: add `<repo>/<repo>_to_xpolicylab.yml` (+ script) writing the same layout and manifest.
