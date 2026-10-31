# Quickstart

Connect to the stack from your laptop or a cluster node. You need the account details you were given: your name, a Keycloak password, an MLflow token and an S3 key pair; to push logs, the Loki push token. The served models need no token.

## 1. Export the environment

Run this in your clone of this repo (add it to `~/.bashrc` or `~/.zshrc` to keep it), after filling in the five `FLYWHEEL_*` values at its top with your account details; the rest reads them. The host settings come from `services/.env.example`. An admin can print this block with your details and the host filled in: `services/ctl.sh user env <name>` (`user add` prints it with your new secrets).

```bash
# Your account: fill in these five with the details you were given (empty: that service is left unconfigured)
export FLYWHEEL_USER=           # your name
export FLYWHEEL_S3_SECRET=      # your S3 secret
export FLYWHEEL_MLFLOW_TOKEN=   # your MLflow token
export FLYWHEEL_GITEA_TOKEN=    # your Gitea token
export FLYWHEEL_LOKI_TOKEN=     # the Loki push token, only if you push logs

eval "$(grep -E '^(SERVICE_NODE|SERVICE_HOST|CADDY_PORT)=' services/.env.example | sed 's/^/export /')"
export FLYWHEEL_CA=~/.flywheel/ca.crt

# S3 (aws CLI, boto3)
export AWS_ACCESS_KEY_ID=$FLYWHEEL_USER AWS_SECRET_ACCESS_KEY=$FLYWHEEL_S3_SECRET AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL=https://s3.$SERVICE_HOST:$CADDY_PORT AWS_CA_BUNDLE=$FLYWHEEL_CA
# the same buckets as folders (<bucket>/<key>) on the cluster's shared file system, where the S3 gateway keeps them
export FLYWHEEL_BUCKET_PATH=$(sed -n 's/^STATE_DIR=//p' services/.env)/versitygw/buckets

# MLflow (CLI, Python client); large artifacts go through the server, else they hang
export MLFLOW_TRACKING_URI=https://$SERVICE_HOST:$CADDY_PORT/mlflow
export MLFLOW_TRACKING_USERNAME=$FLYWHEEL_USER MLFLOW_TRACKING_PASSWORD=$FLYWHEEL_MLFLOW_TOKEN
export MLFLOW_TRACKING_SERVER_CERT_PATH=$FLYWHEEL_CA
export MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false

# Gitea (pipelines; token from Gitea > Settings > Applications, scope repository read and write)
export GITEA_URL=https://$SERVICE_HOST:$CADDY_PORT/gitea GITEA_TOKEN=$FLYWHEEL_GITEA_TOKEN

# Loki (only if you push logs)
export LOKI_URL=https://$SERVICE_HOST:$CADDY_PORT/loki LOKI_TOKEN=$FLYWHEEL_LOKI_TOKEN

# Triton (image models; no token)
export TRITON_URL=triton.$SERVICE_HOST:$CADDY_PORT

# LLM gateway (the language models; OpenAI-compatible, no token)
export LLM_URL=https://llm.$SERVICE_HOST:$CADDY_PORT/v1
```

## 2. Set up SSH (from a laptop)

The stack listens on one HTTPS port on `SERVICE_NODE`, reachable through a login node. Skip this section on the cluster itself.

1. Add a host to `~/.ssh/config` that forwards the port (fill in the login node, your user and the values from step 1):
   ```
   Host flywheel
     HostName <login-node>
     User <your cluster user>
     LocalForward 8443 <SERVICE_NODE>:8443
     # a keepalive every 30 s, and give up after 3 unanswered (~90 s), so a dropped tunnel exits instead of hanging
     ServerAliveInterval 30
     ServerAliveCountMax 3
   ```
2. Point the stack's names at the tunnel (once):
   ```bash
   echo "127.0.0.1 $SERVICE_HOST fiftyone.$SERVICE_HOST rerun.$SERVICE_HOST s3.$SERVICE_HOST triton.$SERVICE_HOST llm.$SERVICE_HOST" | sudo tee -a /etc/hosts
   ```
3. Open the tunnel whenever you use the stack:
   ```bash
   ssh -N flywheel
   ```

## 3. Trust the stack's certificate

1. Download it for the command-line tools (step 1 points them at it):
   ```bash
   mkdir -p ~/.flywheel && curl -k https://$SERVICE_HOST:$CADDY_PORT/ca.crt -o $FLYWHEEL_CA
   ```
2. Add it to the system trust store once, so the browser trusts the stack too:
   ```bash
   sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain $FLYWHEEL_CA   # macOS
   sudo cp $FLYWHEEL_CA /usr/local/share/ca-certificates/flywheel-ca.crt && sudo update-ca-certificates   # Ubuntu
   ```

## 4. Check the connection

Open `https://$SERVICE_HOST:$CADDY_PORT/` and sign in with your name and Keycloak password; one sign-in covers every web app, and the page links to all of them.

![Landing page](assets/quickstart/landing.png)

## 5. Use the stack from a shell

Install the command-line tools once, with [uv](https://docs.astral.sh/uv/): each tool gets its own environment and lands on your `PATH`, nothing goes into your system Python.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # uv itself, if `uv --version` fails; then open a new shell
uv tool install awscli                             # aws: the S3 buckets
uv tool install mlflow-skinny                      # mlflow: runs and the model registry
uv tool install rerun-sdk==0.41.0                  # rerun: the desktop viewer at the stack's version, for large bags (services/rerun)
brew install tea                                   # Gitea's CLI; on Linux, take the binary from https://dl.gitea.com/tea/
```

`uv tool upgrade --all` updates them, except the pinned `rerun-sdk`, which should stay at the viewer's version. For a one-off command without installing, use `uvx`, e.g. `uvx --from awscli aws s3 ls s3://raw/`.

### S3: data in the buckets

`raw` holds data as collected, `processed` what the pipelines made of it ([more](services/versitygw/README.md)).

```bash
aws s3 ls s3://raw/                                              # list (add --recursive for every object)
aws s3 sync ./capture s3://raw/internal_datasets/<dataset>/      # upload a folder; reruns send only what changed
aws s3 sync s3://processed/<prefix>/ ./<prefix>/                 # download a folder
aws s3 cp ./file s3://raw/<dest>/file                            # one file
```

### MLflow: runs and models

```bash
mlflow experiments search                                        # experiments and their IDs
mlflow runs list --experiment-id <id>                            # runs of one experiment
uv run --with mlflow-skinny python -c 'import mlflow; [print(m.name, dict(m.aliases)) for m in mlflow.MlflowClient().search_registered_models()]'   # the registry
mlflow artifacts download --artifact-uri models:/<model>/<version> --dst-path ./<model>    # a registered model, into ./<model>
mlflow artifacts download --artifact-uri models:/<model>@<alias> --dst-path ./<model>      # by alias
mlflow artifacts download --artifact-uri runs:/<run id>/<path> --dst-path ./<run id>       # a run's artifacts, into ./<run id>
```

Registering, aliasing and logging go through the Python client (`mlflow.MlflowClient()`; in a script, `uv run --with mlflow-skinny python script.py`), which reads the same variables; see [mlflow/](services/mlflow/README.md) and, to serve a model, [triton/](services/triton/README.md) (what Dynamo and Triton serve is set in the Gitea repos `dynamo` and `triton`, not by alias).

### Gitea: pipelines

The workflows live in the private repo `admin/pipelines`; ask an admin to add you as a collaborator ([more](services/gitea/README.md)).

```bash
tea login add --name flywheel --url $GITEA_URL --token $GITEA_TOKEN     # once
tea repos ls --login flywheel
git -c http.sslCAInfo=$FLYWHEEL_CA -c http.extraHeader="Authorization: token $GITEA_TOKEN" \
  clone $GITEA_URL/admin/pipelines.git
```

`tea` has no Actions commands; start and follow workflows through the API:

```bash
gitea() { curl -s --cacert $FLYWHEEL_CA -H "Authorization: token $GITEA_TOKEN" -H 'Content-Type: application/json' "$GITEA_URL/api/v1/repos/admin/pipelines/actions/$1" "${@:2}"; }
gitea workflows | python3 -c 'import json,sys; [print(w["path"].split("/")[-1]) for w in json.load(sys.stdin)["workflows"]]'
gitea workflows/sync-fiftyone-raw.yml/dispatches -d '{"ref":"main","inputs":{"datasets":"internal_datasets/*"}}'
gitea runs | python3 -c 'import json,sys; [print(r["id"], r["status"], r["conclusion"], r["path"].split("@")[0]) for r in json.load(sys.stdin)["workflow_runs"][:10]]'
gitea runs/<run id>/jobs | python3 -c 'import json,sys; [print(j["id"], j["name"], j["status"], j["conclusion"]) for j in json.load(sys.stdin)["jobs"]]'
gitea jobs/<job id>/logs | tail -50
```

Each workflow's inputs are listed in its header ([stages](services/gitea/README.md)).

### Logs: Loki

Push lines with labels you will filter on ([more](services/loki/README.md)); expect HTTP 204:

```bash
curl -s -o /dev/null -w '%{http_code}\n' --cacert $FLYWHEEL_CA -H "Authorization: Bearer $LOKI_TOKEN" \
  -H 'Content-Type: application/json' -X POST $LOKI_URL/api/v1/push --data-raw \
  "{\"streams\":[{\"stream\":{\"job\":\"robot\",\"host\":\"$(hostname)\"},\"values\":[[\"$(date +%s%N)\",\"hello\"]]}]}"
```

Read them in Grafana (`https://$SERVICE_HOST:$CADDY_PORT/grafana`): **Explore**, data source Loki, query `{job="robot"}`. Queries need the SSO login, so there is no token for reading from a shell. GPU, CPU and memory use, free GPUs and who uses them are on the dashboard **Flywheel → Cluster usage** ([prometheus/](services/prometheus/README.md)).

### Triton: served models

```bash
triton() { curl -s --cacert $FLYWHEEL_CA "https://$TRITON_URL/v2/$1" "${@:2}"; }
triton health/ready -o /dev/null -w '%{http_code}\n'               # 200 when up
triton repository/index -X POST                                   # models and their state (image models, policies)
triton models/<model>                                             # its inputs and outputs
```

Calling a model from Python: [triton/](services/triton/README.md#example-usage).

### LLMs: the gateway

```bash
curl -s --cacert $FLYWHEEL_CA $LLM_URL/models                     # the models served now; their ids go in "model"
curl -s --cacert $FLYWHEEL_CA $LLM_URL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "<model>", "messages": [{"role": "user", "content": "Hello"}]}'
```

What is served, and calls from Python, with images or marked mission-critical: [dynamo/](services/dynamo/README.md#use).
