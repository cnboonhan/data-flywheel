# rerun

Rerun web viewer (`Dockerfile`, `rerun --serve-web`) at `https://rerun.$SERVICE_HOST:$CADDY_PORT/`, behind SSO. Recordings are served from the same origin: `/data/` = `processed/rerun/` (`ingest-episodes` writes one `.rrd` per episode, see [fiftyone/](../fiftyone/README.md)).

> **Clean setup in progress (2026-10-08).** Only the ingest workflows (`download-datasets-hf`, `download-models-hf` in `services/gitea/ingest/`) exist now. The processing, verify, train and evaluate workflows named below were removed and will be rebuilt under `services/gitea/{clean,validate,mix}/`; the scripts they called (`services/fiftyone/`, `services/xpolicylab/`, `services/slurm/`) are still here.

```bash
# open a recording (this is the rerun_url field on each FiftyOne sample)
https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/data/<dataset>/<episode_id>.rrd
# put any other recording where the viewer can fetch it
aws --endpoint-url https://s3.$SERVICE_HOST:$CADDY_PORT s3 cp capture.rrd s3://processed/rerun/<dataset>/capture.rrd
# inspect raw data locally without converting
rerun --save out.rrd <bag.mcap>
```

The viewer's gRPC port is not exposed.
