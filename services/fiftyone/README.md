# fiftyone

FiftyOne App at `https://fiftyone.$SERVICE_HOST:$CADDY_PORT/` (SSO), over MongoDB (`mongo` service), showing the `raw` bucket. Open it in Chrome: Safari fails to load MCAP streams.

The Gitea workflow [`sync-fiftyone-raw`](../gitea/ingest/sync-fiftyone-raw.yml) mirrors `raw` every 30 minutes: each `raw/<group>/<name>/` becomes the dataset `raw/<group>/<name>`, with samples pointing at the raw files (nothing is converted or copied).

| Raw layout | Dataset |
|---|---|
| LeRobot v3 (`meta/info.json`) | one sample per episode, a media reference into the source videos and parquet |
| xspark (`*/data/episode_*.hdf5`) | one group per episode, a slice per `preview_video/<episode>_<camera>.mp4` |
| COLMAP (`images/` beside `sparse/0`, text or binary) | one image sample per photo with its pose (`position`, `quaternion_wxyz`), camera and intrinsics; e.g. `sensors/real2sim` captures |
| `*.mcap`, `*.bag` | one sample per bag, with rosbag2 `metadata.yaml` fields and `rerun_url`, which opens the bag in [Rerun](../rerun/README.md) |
| `*.tar`, `*.tar.gz`, `*.zip` | one sample per archive, catalog only |

## Use

1. To see new data sooner, dispatch `sync-fiftyone-raw` with input `datasets`, a glob such as `internal_datasets/*` ([how to dispatch](../gitea/README.md)).
2. Check what training will inherit: episode count matches the collection log, every episode has all cameras, durations are plausible, task strings are right.
3. Debug the App or the database:
   ```bash
   services/ctl.sh logs -f fiftyone
   docker exec flywheel-mongo-1 mongosh --quiet fiftyone --eval 'db.datasets.find({},{name:1,_id:0})'
   ```

## Notes

- Incremental: re-runs touch only files that are new, changed or gone, tracked by the `raw_unit` and `raw_sig` fields; a dataset whose folder disappears is deleted.
- The image is built from `Dockerfile` (stock image plus protobuf, pyarrow, h5py); the sync job uses the same image. `VFF_MULTIMODAL` is on, so the App plays LeRobot episode references and MCAP bags.
- Media is read from `/buckets`, the buckets mounted read-only; Actions jobs see the same path.
- The App is one shared instance: people browsing at the same time can move each other's view.
- First full sync (2026-10-08): HiFi-UMI-2K 482 060 episodes, robodojo 6 257 camera slices (2 099 episodes), h2rc 4 121 bags, galaxea 227 archives.
