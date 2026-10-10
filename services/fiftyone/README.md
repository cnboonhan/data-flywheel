# fiftyone

FiftyOne App (`https://fiftyone.$SERVICE_HOST:$CADDY_PORT/`, behind SSO) over MongoDB (`mongo` service), built from `Dockerfile` (stock image plus protobuf, pyarrow, h5py; the sync job uses the same image). Media is read from `/buckets`, the buckets mounted read-only; Actions jobs see the same path. `VFF_MULTIMODAL` is on, so the App plays LeRobot episode references and MCAP bags.

The Gitea workflow [`sync-fiftyone-raw`](../gitea/ingest/sync-fiftyone-raw.yml) mirrors the raw bucket every 30 minutes. Each `raw/<group>/<name>/` becomes the dataset `raw/<group>/<name>`. Samples point at the raw files, so nothing is converted or copied. Re-runs touch only files that are new, changed or gone, tracked by the `raw_unit` and `raw_sig` fields.

| Raw layout | Dataset |
|---|---|
| LeRobot v3 (`meta/info.json`) | one sample per episode, a media reference into the source videos and parquet |
| xspark (`*/data/episode_*.hdf5`) | one group per episode, a slice per `preview_video/<episode>_<camera>.mp4` |
| COLMAP (`images/` beside `sparse/0`, text or binary) | one image sample per photo with its pose (`position`, `quaternion_wxyz`), camera and intrinsics; e.g. `sensors/real2sim` captures |
| `*.mcap`, `*.bag` | one sample per bag, with rosbag2 `metadata.yaml` fields and `rerun_url`, which opens the bag in [Rerun](../rerun/README.md) |
| `*.tar`, `*.tar.gz`, `*.zip` | one sample per archive, catalog only |

Use Chrome: Safari fails to load MCAP streams. The App is one shared instance: people browsing at the same time can move each other's view. Dispatch the workflow by hand (input `datasets`, a glob such as `internal_datasets/*`) to see new data sooner.

What to check, because training inherits it: episode count matches the collection log, every episode has all cameras, durations are plausible, task strings are right.

First full sync (2026-10-08): HiFi-UMI-2K 482 060 episodes, robodojo 6 257 camera slices (2 099 episodes), h2rc 4 121 bags, galaxea 227 archives.

```bash
services/ctl.sh logs -f fiftyone
docker exec flywheel-mongo-1 mongosh --quiet fiftyone --eval 'db.datasets.find({},{name:1,_id:0})'
```
