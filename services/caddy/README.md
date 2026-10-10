# caddy

TLS termination, routing and the local CA, all on `$CADDY_PORT`. Apps that work under a path are routed on `$SERVICE_HOST`; the three that don't get a subdomain.

| URL | Upstream |
|---|---|
| `/` | landing page (`site/index.html`) |
| `/gitea`, `/mlflow`, `/grafana`, `/loki` | the service |
| `/auth`, `/oauth2` | Keycloak, oauth2-proxy |
| `/ca.crt` | root certificate of the local CA |
| `https://fiftyone.$SERVICE_HOST` | FiftyOne |
| `https://rerun.$SERVICE_HOST` | Rerun ([rerun/](../rerun/README.md)) |
| `https://s3.$SERVICE_HOST` (`/ui/` behind SSO) | Versity S3 gateway |
| `https://triton.$SERVICE_HOST` | Triton, bearer token ([triton/](../triton/README.md)) |

1. Apply an edit to `Caddyfile` or `site/`.
   ```bash
   services/ctl.sh restart caddy
   ```
2. Fetch the CA root for a client: [Access](../README.md#access).

**Notes**
- `/loki`, FiftyOne, Rerun and the S3 UI are gated by the `(sso)` snippet (forward auth to oauth2-proxy). Exception: `POST /loki/api/v1/push` with the `LOKI_PUSH_TOKEN` bearer token ([loki/](../loki/README.md)).
- `$SERVICE_HOST` must resolve to the node from every client; `flywheel.<ip>.sslip.io` works without DNS.
- CA material is in `$STATE_DIR/caddy/data/caddy/pki/authorities/local/`. Gitea and oauth2-proxy mount `root.crt`, so `ctl.sh up` starts Caddy first on a clean state.
- A subdomain page that spins forever means the name didn't go through the tunnel: add it to `/etc/hosts` next to `$SERVICE_HOST`.
