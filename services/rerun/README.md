# rerun

Rerun web viewer (`rerun --serve-web`, built from `Dockerfile`) at `https://rerun.$SERVICE_HOST:$CADDY_PORT/` (SSO). The same origin serves recordings from `processed/rerun/` at `/data/` and the `raw` bucket, read-only, at `/raw/`.

Open a recording:

```
https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/data/<dataset>/<episode_id>.rrd
```

Open a raw bag (FiftyOne's MCAP samples carry this link as `rerun_url`):

```
https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/raw/$RERUN_RAW_TOKEN/<group>/<dataset>/<path>.mcap
```

Put any other recording where the viewer can fetch it ([S3 client](../versitygw/README.md#client-setup)):

```bash
aws s3 cp capture.rrd s3://processed/rerun/<dataset>/capture.rrd
```

Inspect a bag locally without converting it:

```bash
rerun --save out.rrd <bag.mcap>
```

## Notes

- The viewer fetches files without a login cookie, so bags (`.mcap`, `.bag` only) are also served without login under `/raw/<RERUN_RAW_TOKEN>/` (token in `services/.env`). Anyone with a link can read the bag; changing the token revokes every link.
- The viewer's gRPC port is not exposed.
