# versitygw

Versity S3 gateway over plain directories: `$STATE_DIR/versitygw/buckets/<bucket>/` is the bucket, every file an object. API at `https://s3.$SERVICE_HOST:$CADDY_PORT/`, web UI at `/ui/` (SSO).

| Bucket | Holds | S3 access |
|---|---|---|
| `raw` | collected data as it arrived | every gateway user, read/write |
| `processed` | converted datasets, splats, recordings | every gateway user, read/write |
| `mlflow` | MLflow artifacts and checkpoints | root key only |
| `triton` | Triton's model repository and sync records | root key only |

## Client setup

1. Get keys: `ctl.sh user add` ([Users](../README.md#users)).
2. Fetch the CA root once (through a login node, set up the [port forward](../README.md#access) first).
   ```bash
   mkdir -p ~/.aws && curl -k https://$SERVICE_HOST:$CADDY_PORT/ca.crt -o ~/.aws/flywheel-ca.crt
   ```
   On cluster nodes, use `$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt` instead; `slurm.env` already sets `AWS_CA_BUNDLE` to it.
3. Configure the client, either as a profile:
   ```ini
   # ~/.aws/config
   [profile flywheel]
   region = us-east-1
   endpoint_url = https://s3.<SERVICE_HOST>:<CADDY_PORT>
   ca_bundle = /home/<you>/.aws/flywheel-ca.crt

   # ~/.aws/credentials (chmod 600)
   [flywheel]
   aws_access_key_id = <key>
   aws_secret_access_key = <secret>
   ```
   ```bash
   export AWS_PROFILE=flywheel
   ```
   or as environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=<key> AWS_SECRET_ACCESS_KEY=<secret> AWS_DEFAULT_REGION=us-east-1 \
          AWS_ENDPOINT_URL=https://s3.$SERVICE_HOST:$CADDY_PORT AWS_CA_BUNDLE=$HOME/.aws/flywheel-ca.crt
   ```
4. Run the CLI without installing it, or install it once.
   ```bash
   uvx --from awscli aws s3 ls s3://raw/      # or: uv tool install awscli
   ```

The aws CLI and boto3 (>= 1.28) read both forms, so repo scripts such as `sim/splat/train.py` need nothing more (their `S3_ENDPOINT_URL`, if set, takes precedence).

## Copy folders

```bash
aws s3 ls s3://processed/                                        # list (add --recursive for every object)
aws s3 sync ./capture s3://raw/internal_datasets/<dataset>/      # upload a folder; reruns send only new or changed files
aws s3 sync s3://processed/real2sim/<run>/ ./<run>/              # download a folder
aws s3 sync ./capture s3://raw/<dest>/ --exclude '*' --include '*.jpg' --dryrun   # filter; --dryrun shows the plan
aws s3 cp ./file s3://raw/<dest>/file                            # one file (cp --recursive copies a folder unconditionally)
aws s3 rm s3://raw/<dest>/ --recursive                           # delete a prefix
```

On the service node, move a directory into a bucket instead of copying it (instant; the objects then have no ETag):

```bash
mv <dir> $STATE_DIR/versitygw/buckets/raw/internal_datasets/<dataset>
```

## Notes

- `sync` copies the folder's contents into the target prefix, not a subfolder named after the source: name the target folder in the destination. `--delete` removes target files that are gone from the source.
- **Access control.** `ctl.sh up` and `user add` regenerate the `raw` and `processed` bucket policies from the gateway's user list (`apply_s3_policies`, `S3_SHARED_BUCKETS` in `ctl.sh`); `mlflow` and `triton` have no policy, so only the root key (`ADMIN_USER` / `ADMIN_PASSWORD`) reaches them. A policy can also scope a user to a key prefix (`Resource: arn:aws:s3:::raw/<prefix>/*`, listing via an `s3:prefix` condition).
- Policies govern S3 requests only. FiftyOne, Rerun and the Slurm jobs read the bucket directories on disk directly.
- `meta/` holds the object metadata in sidecar files (`/tier1` has no xattrs), `iam/` the users. The gateway deletes a folder once its last object is gone.
- `ctl.sh up` creates the buckets `raw`, `processed`, `mlflow` and `triton/models`.
