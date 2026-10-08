# rerun

Rerun web viewer (`Dockerfile`, `rerun --serve-web`) at `https://rerun.$SERVICE_HOST:$CADDY_PORT/`, behind SSO. Served from the same origin, behind the same login: `/data/` = `processed/rerun/` (recordings) and `/raw/` = the `raw` bucket, read-only. The viewer imports raw files it can read, such as MCAP bags, straight from the bucket. Its fetch sends no login cookie, so bags are also served without login under `/raw/<RERUN_RAW_TOKEN>/` (`.mcap` and `.bag` only, token in `.env`). That is the `rerun_url` on FiftyOne's MCAP samples; changing the token revokes every link.

```bash
# open a recording
https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/data/<dataset>/<episode_id>.rrd
# open a raw bag
https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/raw/$RERUN_RAW_TOKEN/<group>/<dataset>/<path>.mcap
# put any other recording where the viewer can fetch it
aws --endpoint-url https://s3.$SERVICE_HOST:$CADDY_PORT s3 cp capture.rrd s3://processed/rerun/<dataset>/capture.rrd
# inspect raw data locally without converting
rerun --save out.rrd <bag.mcap>
```

The viewer's gRPC port is not exposed.
