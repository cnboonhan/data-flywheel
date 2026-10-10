# versitygw

Versity S3 gateway over plain directories: `$STATE_DIR/versitygw/buckets/<bucket>/` is the bucket, every file an object (`meta/` holds the sidecar metadata, `iam/` the users; `/tier1` has no xattrs). API at `https://s3.$SERVICE_HOST:$CADDY_PORT/`, web UI at `/ui/` behind SSO. Root key = `ADMIN_USER` / `ADMIN_PASSWORD`; per-user keys from `ctl.sh user add`. Buckets `raw`, `processed`, `mlflow` are created by `ctl.sh up`.

## Client setup

Keys from `ctl.sh user add`; `flywheel-ca.crt` from `https://$SERVICE_HOST:$CADDY_PORT/ca.crt` ([Access](../README.md#access)).
The CLI: `uvx --from awscli aws ...` (no install), or `uv tool install awscli`.

Either a profile, once per machine:

```ini
# ~/.aws/config
[profile flywheel]
region = us-east-1
endpoint_url = https://s3.<SERVICE_HOST>:<CADDY_PORT>
ca_bundle = /abs/path/flywheel-ca.crt

# ~/.aws/credentials (chmod 600)
[flywheel]
aws_access_key_id = <key>
aws_secret_access_key = <secret>
```

then `export AWS_PROFILE=flywheel`. Or environment variables only:

```bash
export AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret> AWS_DEFAULT_REGION=us-east-1 \
       AWS_ENDPOINT_URL=https://s3.$SERVICE_HOST:$CADDY_PORT AWS_CA_BUNDLE=/abs/path/flywheel-ca.crt
```

The aws CLI and boto3 (>= 1.28) read both, so repo scripts such as `sim/splat/train.py` need nothing more
(their `S3_ENDPOINT_URL`, if set, takes precedence).

## Copy folders

```bash
aws s3 ls s3://processed/                                        # list (add --recursive for every object)
aws s3 sync ./capture s3://raw/internal_datasets/<dataset>/      # upload a folder; reruns send only new or changed files
aws s3 sync s3://processed/real2sim/<run>/ ./<run>/              # download a folder
aws s3 sync ./capture s3://raw/<dest>/ --exclude '*' --include '*.jpg' --dryrun   # filter; --dryrun shows the plan
aws s3 cp ./file s3://raw/<dest>/file                            # one file (cp --recursive copies a folder unconditionally)
aws s3 rm s3://raw/<dest>/ --recursive                           # delete a prefix
```

`sync` copies the folder's contents into the target prefix (it does not create a subfolder named after the source),
so name the target folder in the destination. Add `--delete` to remove target files that are gone from the source.

On the service node, moving a directory into the bucket is instant and needs no copy (the objects then have no ETag):

```bash
mv <dir> $STATE_DIR/versitygw/buckets/raw/internal_datasets/<dataset>
```

Data on disk stays readable as files: FiftyOne, Rerun and the Slurm jobs read the bucket directories directly.
