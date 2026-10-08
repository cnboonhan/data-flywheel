# adapter

Open datasets → XPolicyLab xspark at `s3://processed/xpolicylab/<bench>/<task>/<env_cfg>/data/episode_%07d.hdf5`. Incremental: a per-task `manifest.json` maps source → output episodes, so re-runs only add what's new.

| Folder | Source → bench |
|---|---|
| [`galaxeaOpenWorldDataset/`](galaxeaOpenWorldDataset/) | `raw/open_datasets/galaxea-open-world-r1lite/lerobot/<task>.tar.gz` → `galaxeaOpenWorldDataset` |
| [`robodojo/`](robodojo/) | `raw/open_datasets/robodojo/<task>/<env_cfg>/data/` (copy) → `RoboDojo` |

Inputs: `tasks` (glob, default `*`), `limit` (episodes per task, 0 = all), `env_cfg` (default `arx_x5`).

New source: add `<repo>/<repo>_to_xpolicylab.yml` (+ script) writing the same layout and manifest.
