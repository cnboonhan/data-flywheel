# fiftyone

FiftyOne App (`https://fiftyone.$SERVICE_HOST:$CADDY_PORT/`, behind SSO) over MongoDB (`mongo` service), media streamed from `/buckets` (read-only mount of the buckets, same path in Actions jobs). The scripts here run as Actions jobs ([gitea/](../gitea/README.md)).

> **Clean setup in progress (2026-10-08).** Only the ingest workflows (`download-datasets-hf`, `download-models-hf` in `services/gitea/ingest/`) exist now. The processing, verify, train and evaluate workflows named below were removed and will be rebuilt under `services/gitea/{clean,validate,mix}/`; the scripts they called (`services/fiftyone/`, `services/xpolicylab/`, `services/slurm/`) are still here.

**Canonical episodes**, one layout for every raw format, in the `processed` bucket:

```
processed/episodes/<dataset>/<...>/<episode_id>/
    <camera>.mp4        one per camera, common clock
    episode.json        dataset, episode_id, source, format, robot, task, tasks, fps, frames, duration_s, cameras{...}
    signals.parquet     t, group, index, value
```

| Script | Workflow | Input |
|---|---|---|
| `episodes_from_lerobot.py` | `episodes-lerobot` | LeRobot v2 (per-episode mp4) and v3 (episodes cut from per-camera mp4s) |
| `episodes_from_mcap.py` | `episodes-mcap` | ROS 2 bags: `CompressedImage` → mp4, JointState/IMU/Wrench → signals |
| `episodes_from_xspark.py` | `episodes-xspark` | xspark HDF5 (RoboDojo); also copies the training tree |
| `episodes.py` | | shared layout and S3 writers |
| `episode_rrd.py` + `ingest_episodes.py` | `ingest-episodes` | episodes → `processed/rerun/<dataset>/<episode_id>.rrd` + grouped dataset `episodes/<dataset>` (one group per episode, one slice per camera, `rerun_url`) |
| `unpack_archives.py` | `unpack-archives` | tar archives `raw` → `processed` (`.unpacked` marker) |
| `ingest_lerobot.py`, `convert_mcap.py`, `ingest_videos.py` | `ingest-lerobot`, `convert-mcap` | earlier per-format ingests, superseded |

Jobs in the FiftyOne image fetch the repo archive from Gitea's API (no `node` for `actions/checkout`). `RERUN_BASE` (repo variable) sets the `rerun_url` prefix.

```bash
services/ctl.sh logs -f fiftyone
docker exec flywheel-mongo-1 mongosh --quiet fiftyone --eval 'db.datasets.find({},{name:1,_id:0})'
```
