# caddy

TLS termination, routing and the local CA, all on `$CADDY_PORT`. Apps that work under a path are routed on `$SERVICE_HOST`; the three that don't get a subdomain.

| URL | Upstream |
|---|---|
| `/` | landing page (`site/index.html`); hovering a link pops up a random bufo from the `assets/bufo` submodule, served at `/bufo/` |
| `/gitea`, `/mlflow`, `/grafana`, `/loki` | the service |
| `/auth`, `/oauth2` | Keycloak, oauth2-proxy |
| `/ca.crt` | root certificate of the local CA |
| `https://fiftyone.$SERVICE_HOST` | FiftyOne |
| `https://rerun.$SERVICE_HOST` | Rerun ([rerun/](../rerun/README.md)) |
| `https://s3.$SERVICE_HOST` (`/ui/` behind SSO) | Versity S3 gateway |
| `https://triton.$SERVICE_HOST` | Triton, bearer token ([triton/](../triton/README.md)) |

1. After editing `Caddyfile` or `site/`, restart Caddy.
   ```bash
   services/ctl.sh restart caddy
   ```
2. To trust the CA on a client, follow [Access](../README.md#access).

**Notes**
- To put a new app behind the login, import the `(sso)` snippet in its route, as `/loki`, FiftyOne, Rerun and the S3 UI do (forward auth to oauth2-proxy). `POST /loki/api/v1/push` is the exception: it takes the `LOKI_PUSH_TOKEN` bearer token instead ([loki/](../loki/README.md)).
- Set `$SERVICE_HOST` to a name every client resolves to the node; `flywheel.<ip>.sslip.io` works without DNS.
- Find the CA files in `$STATE_DIR/caddy/data/caddy/pki/authorities/local/`. Gitea and oauth2-proxy mount `root.crt`, which is why `ctl.sh up` starts Caddy first on a clean state; keep that order if you change the startup.
- If a subdomain page spins forever, the name isn't going through your tunnel: add it to `/etc/hosts` next to `$SERVICE_HOST`.
