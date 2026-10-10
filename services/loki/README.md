# loki

Log store (config `loki.yml`, data in `$STATE_DIR/loki`). No UI of its own: read logs in [Grafana](../grafana/README.md). Nothing ships logs automatically yet.

| Route | Access |
|---|---|
| `POST https://$SERVICE_HOST:$CADDY_PORT/loki/api/v1/push` | bearer token `LOKI_PUSH_TOKEN` (`services/.env`); an empty token disables the route |
| everything else under `/loki` (queries, labels) | SSO login |
| `http://loki:3100` | no auth, compose network only |

## Push logs

Fetch the CA root once ([Access](../README.md#access)), then push with the token from `services/.env`:

```bash
TOKEN=<LOKI_PUSH_TOKEN from services/.env>
curl --cacert flywheel-ca.crt -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -X POST https://$SERVICE_HOST:$CADDY_PORT/loki/api/v1/push --data-raw \
  "{\"streams\":[{\"stream\":{\"job\":\"robot\",\"host\":\"$(hostname)\"},\"values\":[[\"$(date +%s%N)\",\"hello\"]]}]}"
# Expect 204. A 302 means the token is wrong or missing (you were sent to the login page).
```

- Put the labels you filter on in `stream` (`job`, `host`, ...), and keep them few and low-cardinality.
- Send `values` as `[timestamp in ns as a string, line]` pairs; batch many into one request.
- To ship logs continuously, point Grafana Alloy, Promtail or Vector at the same endpoint with the same bearer token.

## Read logs

In Grafana, open Explore, choose the Loki data source and query:

```logql
{job="robot"}
{job="robot", host="r1-01"} |= "error"
```

Or query from a shell:

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

- To change what the token allows, edit the `@loki_push` matcher in [`../caddy/Caddyfile`](../caddy/Caddyfile). It admits only `POST /loki/api/v1/push` with the bearer token and strips the header before Loki.
