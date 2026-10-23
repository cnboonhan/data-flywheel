# rerun

Rerun web viewer (`rerun --serve-web`, built from `Dockerfile`) at `https://rerun.$SERVICE_HOST:$CADDY_PORT/`, behind SSO. The same origin serves the `raw` and `processed` buckets, read-only, at `/raw/` and `/processed/`.

1. Find a dataset, bag or episode in Browse Data ([browse/](../browse/README.md)): its links open episodes here, so you don't build the links below by hand.
2. Open a raw bag. FiftyOne's MCAP samples carry this link as `rerun_url`, so you can also click through from there.
   ```
   https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/raw/$RERUN_RAW_TOKEN/<group>/<dataset>/<path>.mcap
   ```
3. Open a LeRobot episode (v3, or v2.x such as `sensors/autolearn`'s collections). The viewer opens files, not LeRobot folders, so `rerun-lerobot` (`lerobot_recording_server.py`) builds a recording of the episode on the first request (cameras as videos, every `observation.state*`, `state.*` and `action*` feature as plots, the task as text; ~2 s for a 6-camera, 450-frame episode) and serves it from `$STATE_DIR/rerun-cache/` after. FiftyOne's LeRobot samples carry this link as `rerun_url`:
   ```
   https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/lerobot/$RERUN_RAW_TOKEN/<group>/<dataset>/<root>/episode_<n>.rrd
   ```
   For a dataset in the `processed` bucket, use `/lerobot-processed/` in place of `/lerobot/` (and `/processed/` in place of `/raw/` for a bag).
4. Open a bag of 500 MB or more from Browse Data: its "open in Rerun" link opens a page that turns the `.mcap` into a recording the web viewer can hold, then opens it. The web viewer runs in the browser's ~4 GB and only partly loads such a bag as it is; `mcap_recording.py` encodes each camera as H.264 video (~7x smaller than its JPEGs) and keeps joint states, IMUs, wrenches, twists and strings as plots. The page shows the progress; expect 2-3 minutes for 1.5 GB the first time (1.46 GB to 232 MB), then the recording opens at once from the cache. For the original messages, or a ROS 1 `.bag`, use the desktop viewer: download the bag (its file link in Browse Data, or `aws s3 cp`), then open the file; it can't fetch the links above, as it doesn't trust the flywheel CA.
   ```bash
   uv tool install rerun-sdk==0.38.1      # once; the viewer's version (quickstart: tools)
   rerun <bag.mcap>
   ```
5. To inspect a bag on your own machine without uploading it, convert it locally:
   ```bash
   rerun --save out.rrd <bag.mcap>
   ```

## Notes

- The viewer fetches files without a login cookie, so bags (`.mcap`, `.bag` only) are also served without login under `/raw/<RERUN_RAW_TOKEN>/` (token in `services/.env`). Anyone with such a link can read the bag; to revoke every link, change the token, run `ctl.sh up` and rerun `sync-fiftyone-raw`.
- The viewer's gRPC port is not exposed; use the web viewer.
