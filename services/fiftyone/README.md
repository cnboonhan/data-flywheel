# fiftyone

FiftyOne App over MongoDB (`mongo` service), showing the `raw` bucket. The Gitea workflow [`sync-fiftyone-raw`](../gitea/ingest/sync-fiftyone-raw.yml) mirrors `raw` every 30 minutes: each `raw/<group>/<name>/` becomes the dataset `raw/<group>/<name>`, with samples pointing at the raw files (nothing is converted or copied).

| Raw layout | Dataset |
|---|---|
| LeRobot v3 (`meta/info.json`) | one sample per episode, a media reference into the source videos and parquet |
| xspark (`*/data/episode_*.hdf5`) | one group per episode, a slice per `preview_video/<episode>_<camera>.mp4` |
| COLMAP (`images/` beside `sparse/0`, text or binary) | one image sample per photo with its pose (`position`, `quaternion_wxyz`), camera and intrinsics; e.g. `sensors/real2sim` captures |
| `*.mcap`, `*.bag` | one sample per bag, with rosbag2 `metadata.yaml` fields and `rerun_url`, which opens the bag in [Rerun](../rerun/README.md) |
| `*.tar`, `*.tar.gz`, `*.zip` | one sample per archive, catalog only |

## Use

1. Open `https://fiftyone.$SERVICE_HOST:$CADDY_PORT/` in Chrome (Safari fails to load MCAP streams), sign in, and pick a `raw/...` dataset.
2. To see new data sooner than the next scheduled sync, dispatch `sync-fiftyone-raw` with input `datasets`, a glob such as `internal_datasets/*` ([how to dispatch](../gitea/README.md#run-a-workflow)).
3. Before training on a dataset, check what training will inherit: the episode count matches the collection log, every episode has all its cameras, durations are plausible, task strings are right.
4. If the App or the sync misbehaves, read the App's log and list the datasets in the database:
   ```bash
   services/ctl.sh logs -f fiftyone
   docker exec flywheel-mongo-1 mongosh --quiet fiftyone --eval 'db.datasets.find({},{name:1,_id:0})'
   ```

## Notes

- The sync is incremental: it touches only files that are new, changed or gone (tracked by the `raw_unit` and `raw_sig` fields), and deletes a dataset whose folder disappears. To change how a layout is ingested, edit the workflow and push it ([how](../gitea/README.md#change-or-add-a-workflow)).
- The App and the sync job use one image, built from `Dockerfile` (stock image plus protobuf, pyarrow, h5py). Keep `VFF_MULTIMODAL` on in both: it lets the App play LeRobot episode references and MCAP bags.
- Both read media from `/buckets`, the buckets mounted read-only, so sample paths are the same in the App and in Actions jobs.
- The App is one shared instance: people browsing at the same time can move each other's view.
