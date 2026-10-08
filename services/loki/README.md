# loki

Log store. Config `loki.yml`, data in `$STATE_DIR/loki`. No web UI: read logs in Grafana (Explore, data source Loki). Nothing ships logs automatically yet; push them as below.

| Route | Access |
|---|---|
| `POST https://$SERVICE_HOST:$CADDY_PORT/loki/api/v1/push` | bearer token `LOKI_PUSH_TOKEN` (`services/.env`); empty token disables it |
| everything else under `/loki` (queries, labels) | SSO login (oauth2-proxy) |
| `http://loki:3100` | no auth, compose network only |

## Push

```bash
TOKEN=<LOKI_PUSH_TOKEN from services/.env>
curl --cacert flywheel-ca.crt -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -X POST https://$SERVICE_HOST:$CADDY_PORT/loki/api/v1/push --data-raw \
  "{\"streams\":[{\"stream\":{\"job\":\"robot\",\"host\":\"$(hostname)\"},\"values\":[[\"$(date +%s%N)\",\"hello\"]]}]}"
# 204 = stored. 302 = wrong or missing token (sent to the login page).
```

- `stream`: labels to filter on (`job`, `host`, ...). Keep them few and low-cardinality.
- `values`: `[timestamp in ns as a string, line]` pairs; one request can carry many.
- `flywheel-ca.crt`: `curl -k https://$SERVICE_HOST:$CADDY_PORT/ca.crt -o flywheel-ca.crt` (or `-k` while testing). Through the SSH tunnel, map `$SERVICE_HOST` in `/etc/hosts`.
- Shippers (Grafana Alloy, Promtail, Vector) work the same way: Loki push endpoint above plus the bearer token.

## Read

Grafana > Explore, data source Loki:

```logql
{job="robot"}
{job="robot", host="r1-01"} |= "error"
```

From a shell (inside the stack):

```bash
services/ctl.sh exec -T gitea curl -s -G http://loki:3100/loki/api/v1/query_range \
  --data-urlencode 'query={job="robot"}' --data-urlencode "start=$(( $(date +%s) - 3600 ))000000000"
```

## Operate

```bash
services/ctl.sh restart loki     # after editing loki.yml
services/ctl.sh up               # after changing LOKI_PUSH_TOKEN (Caddy reads it)
```

The push route is the `@loki_push` matcher in `../caddy/Caddyfile`: only `POST /loki/api/v1/push` with `Authorization: Bearer $LOKI_PUSH_TOKEN`; the header is stripped before Loki.
