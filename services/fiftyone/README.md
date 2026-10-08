# fiftyone

FiftyOne App (`https://fiftyone.$SERVICE_HOST:$CADDY_PORT/`, behind SSO) over MongoDB (`mongo` service), built from `Dockerfile` (stock image plus protobuf, pyarrow, h5py; the sync job uses the same image). Media is read from `/buckets`, the buckets mounted read-only; Actions jobs see the same path. `VFF_MULTIMODAL` is on, so the App plays LeRobot episode references and MCAP bags.

The Gitea workflow [`sync-fiftyone-raw`](../gitea/ingest/sync-fiftyone-raw.yml) mirrors the raw bucket hourly. Each `raw/<group>/<name>/` becomes the dataset `raw/<group>/<name>`. Samples point at the raw files, so nothing is converted or copied. Re-runs touch only files that are new, changed or gone, tracked by the `raw_unit` and `raw_sig` fields.

| Raw layout | Dataset |
|---|---|
| LeRobot v3 (`meta/info.json`) | one sample per episode, a media reference into the source videos and parquet |
| xspark (`*/data/episode_*.hdf5`) | one group per episode, a slice per `preview_video/<episode>_<camera>.mp4` |
| `*.mcap`, `*.bag` | one sample per bag, with rosbag2 `metadata.yaml` fields |
| `*.tar`, `*.tar.gz`, `*.zip` | one sample per archive, catalog only |

```bash
services/ctl.sh logs -f fiftyone
docker exec flywheel-mongo-1 mongosh --quiet fiftyone --eval 'db.datasets.find({},{name:1,_id:0})'
```
