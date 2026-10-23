# caddy

TLS termination, routing and the local CA for the whole stack: apps that work under a path are on `$SERVICE_HOST`, the rest (FiftyOne, Rerun, the S3 API, Triton, the LLM gateway) on a subdomain each.

Everything is on one port, at `https://$SERVICE_HOST:$CADDY_PORT` (call it `$HOST`). One Keycloak sign-in covers every page marked SSO.

| Open | What's there | Sign in |
|---|---|---|
| `$HOST/` | the landing page: every app, and the models served now with their load | none |
| `$HOST/data/` | Browse Data: the `raw` and `processed` buckets as folders, LeRobot dataset pages ([browse/](../browse/README.md)) | SSO |
| `$HOST/gitea/` | pipelines and their runs ([gitea/](../gitea/README.md)) | SSO |
| `$HOST/mlflow/` | experiments and the model registry ([mlflow/](../mlflow/README.md)) | SSO; programs use an MLflow token |
| `$HOST/grafana/` | dashboards and log search ([grafana/](../grafana/README.md)) | SSO |
| `$HOST/auth/` | your Keycloak account; admins manage users here | Keycloak |
| `$HOST/ca.crt` | the stack's root certificate, to trust once per device ([Access](../README.md#access)) | none |
| `https://fiftyone.$SERVICE_HOST:$CADDY_PORT/` | FiftyOne: the `raw` datasets, searchable ([fiftyone/](../fiftyone/README.md)) | SSO |
| `https://rerun.$SERVICE_HOST:$CADDY_PORT/` | the Rerun viewer for bags and episodes ([rerun/](../rerun/README.md)) | SSO |
| `https://s3.$SERVICE_HOST:$CADDY_PORT/` | the S3 API, and its web UI at `/ui/` ([versitygw/](../versitygw/README.md)) | API: your S3 keys; UI: SSO |
| `https://triton.$SERVICE_HOST:$CADDY_PORT/` | image models for programs, HTTP and gRPC ([triton/](../triton/README.md)) | none |
| `https://llm.$SERVICE_HOST:$CADDY_PORT/v1/` | language models, OpenAI-compatible ([dynamo/](../dynamo/README.md)) | none; critical traffic: `LLM_CRITICAL_TOKEN` |
| `$HOST/<app>/`, `https://<app>.$SERVICE_HOST:$CADDY_PORT/` | a [local app](#local-apps), listed on the landing page | as its route says |

For programs that push data in:

| Send to | What | Token |
|---|---|---|
| `POST $HOST/loki/api/v1/push` | log lines ([loki/](../loki/README.md)) | `LOKI_PUSH_TOKEN` |
| `PUT`, `POST`, `DELETE $HOST/pushgateway/metrics/job/...` | metrics, e.g. Slurm jobs ([prometheus/](../prometheus/README.md#slurm-jobs)) | `PUSHGATEWAY_TOKEN` |

**Notes** (for operators)
- `/fiftyone`, `/rerun` and `/s3` on `$HOST` redirect to their subdomains; `rerun.…/browse` and `/dataset` redirect to `/data`.
- The landing page is a template (`site/index.html`): its model tables read `/status/models` (the LLM gateway's `/v1/models`) and `/status/load` (one fixed Prometheus query), and nothing else of the gateway or Prometheus is reachable without login. Hovering pops up the architecture diagram (`assets/`, at `/assets/`) or a role page (`assets/devroles/`, included by the template; `/assets/devroles/` itself answers 404).
- Rerun's viewer fetches files without a login cookie, so its links carry the path token `RERUN_RAW_TOKEN` instead (`/raw/<token>/`, `/processed/<token>/` for bags; `/lerobot/<token>/`, `/lerobot-processed/<token>/` for episodes): anyone with such a link can read that file.
- Triton's admin calls (load, unload, settings) are refused here, since `sync-triton` manages its models; Caddy adds Triton's key on the way through. The LLM gateway runs on `DYNAMO_HEAD:8199`.

Caddy's request metrics (per site: rate, latency, requests in flight; plus its CPU) are on `:9180` inside the compose network for [Prometheus](../prometheus/README.md); query them in Grafana's Explore (e.g. `rate(caddy_http_requests_total[5m])`). It is unlikely to be the bottleneck, but if requests in flight keep climbing while the backends' own latency stays flat, or its CPU stays near one core, it is.

1. After editing `Caddyfile` or a [local app](#local-apps)'s route (`site/local/*.caddy`, `*.routes`), check it, then restart Caddy. Edits to `site/`'s templates (`index.html`, `site/local/*.html`) need nothing: Caddy reads them on each request.
   ```bash
   services/ctl.sh run --rm --no-deps caddy caddy validate --config /etc/caddy/Caddyfile
   services/ctl.sh restart caddy
   ```
   After changing Caddy's mounts or environment in `docker-compose.yml`, recreate it instead: `services/ctl.sh up` (it recreates only the containers that changed).
2. To trust the CA on a client, follow [Access](../README.md#access).

**Notes**
- To put a new app behind the login, import the `(sso)` snippet in its route, as `/loki`, `/data`, FiftyOne, Rerun and the S3 UI do (forward auth to oauth2-proxy). `POST /loki/api/v1/push` is the exception: it takes the `LOKI_PUSH_TOKEN` bearer token instead ([loki/](../loki/README.md)).
- Set `$SERVICE_HOST` to a name every client resolves to the node; `flywheel.<ip>.sslip.io` works without DNS.
- Find the CA files in `$STATE_DIR/caddy/data/caddy/pki/authorities/local/`. Gitea and oauth2-proxy mount `root.crt`, which is why `ctl.sh up` starts Caddy first on a clean state; keep that order if you change the startup.
- If a subdomain page spins forever, the name isn't going through your tunnel: add it to `/etc/hosts` next to `$SERVICE_HOST`.

## Local apps

List an app that isn't part of the stack (one running on a cluster node, say) on the landing page, and optionally route it through Caddy behind the login, without committing it: files in `site/local/` but its README are gitignored, and Caddy and the landing page pick them up.

| File in `site/local/` | Does | Needed |
|---|---|---|
| `<app>.html` | a link in the landing page's Local apps section | always |
| `<app>.caddy` | a subdomain `<app>.$SERVICE_HOST` proxied to the app | to route it through Caddy (with the login), for an app that needs no path prefix |
| `<app>.routes` | a path `/<app>/` on the main host proxied to the app | the same, for an app that can serve under a prefix |

1. Add its link, as `site/local/<app>.html` (one list item, a template like `index.html`). For an app with its own reachable URL, this is all; it shows on the next page load.
   ```html
   <li><a href="https://<app>.{{env "SERVICE_HOST"}}:{{env "CADDY_PORT"}}/">App</a> <small>what it is</small></li>
   ```
2. To route it through Caddy, write one route. On a subdomain (safest), as `site/local/<app>.caddy`:
   ```
   https://<app>.{$SERVICE_HOST}:8443 {
   	import sso
   	reverse_proxy <node address>:<port>
   }
   ```
   Or under a path, as `site/local/<app>.routes`:
   ```
   handle /<app>/* {
   	import sso
   	reverse_proxy <node address>:<port>
   }
   ```
   `import sso` puts it behind the Keycloak login; leave it out only for an app with a login of its own. Then validate and restart Caddy (above).
3. For a subdomain, add `<app>.$SERVICE_HOST` to the `/etc/hosts` line of anyone who comes through a tunnel ([Access](../README.md#access)).

Notes on these files (what is served, what to keep out of them): [site/local/](site/local/README.md).
