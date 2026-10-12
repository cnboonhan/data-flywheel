# rerun

Rerun web viewer (`rerun --serve-web`, built from `Dockerfile`) at `https://rerun.$SERVICE_HOST:$CADDY_PORT/`, behind SSO. The same origin serves the `raw` bucket, read-only, at `/raw/`.

1. Open a raw bag. FiftyOne's MCAP samples carry this link as `rerun_url`, so you can also click through from there.
   ```
   https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/raw/$RERUN_RAW_TOKEN/<group>/<dataset>/<path>.mcap
   ```
2. To inspect a bag on your own machine without uploading it, convert it locally:
   ```bash
   rerun --save out.rrd <bag.mcap>
   ```

## Notes

- The viewer fetches files without a login cookie, so bags (`.mcap`, `.bag` only) are also served without login under `/raw/<RERUN_RAW_TOKEN>/` (token in `services/.env`). Anyone with such a link can read the bag; to revoke every link, change the token, run `ctl.sh up` and rerun `sync-fiftyone-raw`.
- The viewer's gRPC port is not exposed; use the web viewer.
