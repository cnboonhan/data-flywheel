# grafana

Dashboards at `/grafana`; Loki data source from `provisioning/`. Login via Keycloak (group `admins` → Admin) or the built-in admin `ADMIN_USER`.

```bash
services/ctl.sh exec grafana grafana cli admin reset-admin-password "$ADMIN_PASSWORD"   # after changing .env
curl -k https://$SERVICE_HOST:$CADDY_PORT/grafana/api/health
```

`ctl.sh up` sets the built-in admin's email to the SSO admin's so the Keycloak login maps onto it.
