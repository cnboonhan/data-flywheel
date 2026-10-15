# Quickstart

Connect to the stack from your laptop or a cluster node. You need the account details you were given: your name, a Keycloak password, an MLflow token and an S3 key pair; to push logs, the Loki push token. The served models need no token.

## 1. Export the environment

Run this in your clone of this repo (add it to `~/.bashrc` or `~/.zshrc` to keep it), replacing the `<...>` values with your account details. The host settings come from `services/.env.example`.

```bash
eval "$(grep -E '^(SERVICE_NODE|SERVICE_HOST|CADDY_PORT)=' services/.env.example | sed 's/^/export /')"
export FLYWHEEL_CA=~/.flywheel/ca.crt

# S3 (aws CLI, boto3)
export AWS_ACCESS_KEY_ID=<your name> AWS_SECRET_ACCESS_KEY=<your S3 secret> AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL=https://s3.$SERVICE_HOST:$CADDY_PORT AWS_CA_BUNDLE=$FLYWHEEL_CA

# MLflow (CLI, Python client); large artifacts go through the server, else they hang
export MLFLOW_TRACKING_URI=https://$SERVICE_HOST:$CADDY_PORT/mlflow
export MLFLOW_TRACKING_USERNAME=<your name> MLFLOW_TRACKING_PASSWORD=<your MLflow token>
export MLFLOW_TRACKING_SERVER_CERT_PATH=$FLYWHEEL_CA
export MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false

# Gitea (pipelines; token from Gitea > Settings > Applications, scope repository read and write)
export GITEA_URL=https://$SERVICE_HOST:$CADDY_PORT/gitea GITEA_TOKEN=<your Gitea token>

# Loki (only if you push logs)
export LOKI_URL=https://$SERVICE_HOST:$CADDY_PORT/loki LOKI_TOKEN=<Loki push token>

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

Install the command-line tools once:

```bash
uv tool install awscli && uv tool install mlflow-skinny
brew install tea          # Gitea's CLI; on Linux, take the binary from https://dl.gitea.com/tea/
```

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
mlflow artifacts download --artifact-uri models:/<model>/<version> --dst-path ./model      # a registered model
mlflow artifacts download --artifact-uri runs:/<run id>/<path> --dst-path ./out            # a run's artifacts
```

Registering, aliasing and logging go through the Python client (`mlflow.MlflowClient()`), which reads the same variables; see [mlflow/](services/mlflow/README.md) and, to serve a model, [triton/](services/triton/README.md).

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

Calling a model from Python: [triton/](services/triton/README.md#call-a-model).

### LLMs: the gateway

```bash
curl -s --cacert $FLYWHEEL_CA $LLM_URL/models
curl -s --cacert $FLYWHEEL_CA $LLM_URL/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3", "messages": [{"role": "user", "content": "Hello"}]}'
```

The first lists the models served now (GLM-5.3 and the Qwen LLMs and VLMs). From Python, with images, and for mission-critical requests: [dynamo/](services/dynamo/README.md#use).
