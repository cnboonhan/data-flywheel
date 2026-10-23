# caddy

TLS termination, routing and the local CA, all on `$CADDY_PORT`. Apps that work under a path are routed on `$SERVICE_HOST`; the three that don't get a subdomain.

| URL | Upstream |
|---|---|
| `/` | landing page (`site/index.html`); hovering a link pops up a random bufo from the `assets/bufo` submodule, served at `/bufo/`; hovering the title shows `site/flywheel.gif`, the line below it `assets/architecture.svg` (repo `assets/` at `/assets/`), and hovering a developer role pops up its [devroles](../../assets/devroles/) page, which the template includes and renders (`/assets/devroles/` itself answers 404: the folder also holds private notes) |
| `/status/models`, `/status/load` | the landing page's model tables: the LLM gateway's `/v1/models` (context window, inputs, licence per model, from `dynamo/models.yaml`), and one fixed Prometheus query (requests, tokens, generation speed and requests in progress per model on Dynamo and Triton, GPU use of their nodes); nothing else of Prometheus or the gateway is reachable without login |
| `/gitea`, `/mlflow`, `/grafana`, `/loki` | the service |
| `/auth`, `/oauth2` | Keycloak, oauth2-proxy |
| `/pushgateway/metrics/job/...` (PUT, POST, DELETE) | Pushgateway, bearer token `PUSHGATEWAY_TOKEN` ([slurm-exporter/](../slurm-exporter/README.md)) |
| `/ca.crt` | root certificate of the local CA |
| `https://fiftyone.$SERVICE_HOST` | FiftyOne |
| `https://rerun.$SERVICE_HOST` | Rerun ([rerun/](../rerun/README.md)); `/browse/` lists the `raw` and `processed` buckets with open-in-Rerun links (`site/rerun-browse.html`) |
| `https://s3.$SERVICE_HOST` (`/ui/` behind SSO) | Versity S3 gateway |
| `https://triton.$SERVICE_HOST` | Triton: inference and metadata open, no token; admin calls (load, unload, settings) refused, as `sync-triton` manages the models ([triton/](../triton/README.md)) |

Caddy's request metrics (per site: rate, latency, requests in flight; plus its CPU) are on `:9180` inside the compose network for [Prometheus](../prometheus/README.md); query them in Grafana's Explore (e.g. `rate(caddy_http_requests_total[5m])`). It is unlikely to be the bottleneck, but if requests in flight keep climbing while the backends' own latency stays flat, or its CPU stays near one core, it is.

1. After editing `Caddyfile` or `site/`, restart Caddy.
   ```bash
   services/ctl.sh restart caddy
   ```
2. To trust the CA on a client, follow [Access](../README.md#access).

## Local apps

Route an app that isn't part of the stack (one running on a cluster node, say) through Caddy without committing it: files in `site/local/` are gitignored, and Caddy and the landing page pick them up.

1. Write the route. On a subdomain of its own (safest: the app needs no path prefix), as `site/local/<app>.caddy`:
   ```
   https://<app>.{$SERVICE_HOST}:8443 {
   	import sso
   	reverse_proxy <node address>:<port>
   }
   ```
   Or under a path of the main host, as `site/local/<app>.routes`, for an app that can serve under a prefix:
   ```
   handle /<app>/* {
   	import sso
   	reverse_proxy <node address>:<port>
   }
   ```
   `import sso` puts it behind the Keycloak login; leave it out only for an app with a login of its own.
2. Add its link to the landing page's Local apps section, as `site/local/<app>.html` (one list item, a template like `index.html`):
   ```html
   <li><a href="https://<app>.{{env "SERVICE_HOST"}}:{{env "CADDY_PORT"}}/">App</a> <small>what it is</small></li>
   ```
3. Restart Caddy (above).
4. For a subdomain, add `<app>.$SERVICE_HOST` to the `/etc/hosts` line of anyone who comes through a tunnel ([Access](../README.md#access)).

**Notes**
- Files in `site/local/` are read by Caddy and the landing page, never served (`/local/*` answers 404).
- Check a route before restarting: `services/ctl.sh run --rm --no-deps caddy caddy validate --config /etc/caddy/Caddyfile`. A broken local file stops Caddy, and every app with it.
- To put a new app behind the login, import the `(sso)` snippet in its route, as `/loki`, FiftyOne, Rerun and the S3 UI do (forward auth to oauth2-proxy). `POST /loki/api/v1/push` is the exception: it takes the `LOKI_PUSH_TOKEN` bearer token instead ([loki/](../loki/README.md)).
- Set `$SERVICE_HOST` to a name every client resolves to the node; `flywheel.<ip>.sslip.io` works without DNS.
- Find the CA files in `$STATE_DIR/caddy/data/caddy/pki/authorities/local/`. Gitea and oauth2-proxy mount `root.crt`, which is why `ctl.sh up` starts Caddy first on a clean state; keep that order if you change the startup.
- If a subdomain page spins forever, the name isn't going through your tunnel: add it to `/etc/hosts` next to `$SERVICE_HOST`.
