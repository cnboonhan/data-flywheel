# caddy

TLS termination, routing and the local CA. Everything listens on `$CADDY_PORT`; apps that can live under a path are routed on `$SERVICE_HOST`, the three that can't get a subdomain.

| URL | Upstream |
|---|---|
| `/` | landing page (`site/index.html`) |
| `/gitea`, `/mlflow`, `/grafana`, `/loki` | the service |
| `/auth`, `/oauth2` | Keycloak, oauth2-proxy |
| `/ca.crt` | root certificate of the local CA |
| `https://fiftyone.$SERVICE_HOST` | FiftyOne |
| `https://rerun.$SERVICE_HOST` (`/data/` = `processed/rerun/`) | Rerun |
| `https://s3.$SERVICE_HOST` (`/ui/` behind SSO) | Versity S3 gateway |

`/loki`, FiftyOne, Rerun and the S3 UI are gated by the `(sso)` snippet (forward auth to oauth2-proxy), except `POST /loki/api/v1/push` with the `LOKI_PUSH_TOKEN` bearer token ([loki/](../loki/README.md)). `$SERVICE_HOST` must resolve to the node from every client; `flywheel.<ip>.sslip.io` does without DNS.

```bash
services/ctl.sh restart caddy                                  # after editing Caddyfile or site/
curl -k https://$SERVICE_HOST:$CADDY_PORT/ca.crt -o flywheel-ca.crt
```

CA material: `$STATE_DIR/caddy/data/caddy/pki/authorities/local/`. Gitea and oauth2-proxy mount `root.crt`, so `ctl.sh up` starts Caddy first on a clean state.

A subdomain page that spins forever means the name did not go through the tunnel: add it to `/etc/hosts` next to `$SERVICE_HOST`.
