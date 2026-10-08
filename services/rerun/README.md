# rerun

Rerun web viewer (`Dockerfile`, `rerun --serve-web`) at `https://rerun.$SERVICE_HOST:$CADDY_PORT/`, behind SSO. Recordings are served from the same origin: `/data/` = `processed/rerun/`.

```bash
# open a recording
https://rerun.$SERVICE_HOST:$CADDY_PORT/?url=https://rerun.$SERVICE_HOST:$CADDY_PORT/data/<dataset>/<episode_id>.rrd
# put any other recording where the viewer can fetch it
aws --endpoint-url https://s3.$SERVICE_HOST:$CADDY_PORT s3 cp capture.rrd s3://processed/rerun/<dataset>/capture.rrd
# inspect raw data locally without converting
rerun --save out.rrd <bag.mcap>
```

The viewer's gRPC port is not exposed.
