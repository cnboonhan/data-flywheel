# versitygw

Versity S3 gateway over plain directories: `$STATE_DIR/versitygw/buckets/<bucket>/` is the bucket, every file an object (`meta/` holds the sidecar metadata, `iam/` the users; `/tier1` has no xattrs). API at `https://s3.$SERVICE_HOST:$CADDY_PORT/`, web UI at `/ui/` behind SSO. Root key = `ADMIN_USER` / `ADMIN_PASSWORD`; per-user keys from `ctl.sh user add`. Buckets `raw`, `processed`, `mlflow` are created by `ctl.sh up`.

```bash
export AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret> AWS_DEFAULT_REGION=us-east-1 AWS_CA_BUNDLE=flywheel-ca.crt
aws --endpoint-url https://s3.$SERVICE_HOST:$CADDY_PORT s3 ls s3://raw/
aws --endpoint-url https://s3.$SERVICE_HOST:$CADDY_PORT s3 sync ./capture s3://raw/internal_datasets/<dataset>/
mv <dir> $STATE_DIR/versitygw/buckets/raw/internal_datasets/<dataset>   # on the node: instant, no copy (objects then have no ETag)
```

Data on disk stays readable as files: FiftyOne, Rerun and the Slurm jobs read the bucket directories directly.
