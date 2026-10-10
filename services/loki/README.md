# loki

Log store (config `loki.yml`, data in `$STATE_DIR/loki`). No UI of its own: read logs in [Grafana](../grafana/README.md). Nothing ships logs automatically yet.

| Route | Access |
|---|---|
| `POST https://$SERVICE_HOST:$CADDY_PORT/loki/api/v1/push` | bearer token `LOKI_PUSH_TOKEN` (`services/.env`); an empty token disables the route |
| everything else under `/loki` (queries, labels) | SSO login |
| `http://loki:3100` | no auth, compose network only |

## Push logs

Fetch the CA root once ([Access](../README.md#access)), then push:

```bash
TOKEN=<LOKI_PUSH_TOKEN from services/.env>
curl --cacert flywheel-ca.crt -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -X POST https://$SERVICE_HOST:$CADDY_PORT/loki/api/v1/push --data-raw \
  "{\"streams\":[{\"stream\":{\"job\":\"robot\",\"host\":\"$(hostname)\"},\"values\":[[\"$(date +%s%N)\",\"hello\"]]}]}"
# 204 = stored. 302 = wrong or missing token (sent to the login page).
```

- `stream`: labels to filter on (`job`, `host`, ...); keep them few and low-cardinality.
- `values`: `[timestamp in ns as a string, line]` pairs; one request can carry many.
- Shippers (Grafana Alloy, Promtail, Vector) use the same endpoint and bearer token.

## Read logs

In Grafana, Explore, data source Loki:

```logql
{job="robot"}
{job="robot", host="r1-01"} |= "error"
```

From a shell:

```bash
services/ctl.sh exec -T gitea curl -s -G http://loki:3100/loki/api/v1/query_range \
  --data-urlencode 'query={job="robot"}' --data-urlencode "start=$(( $(date +%s) - 3600 ))000000000"
```

## Operate

```bash
services/ctl.sh restart loki     # after editing loki.yml
services/ctl.sh up               # after changing LOKI_PUSH_TOKEN (Caddy reads it)
```

## Notes

- The push route is the `@loki_push` matcher in [`../caddy/Caddyfile`](../caddy/Caddyfile): only `POST /loki/api/v1/push` with the bearer token; Caddy strips the header before Loki.
