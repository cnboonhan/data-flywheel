# grafana

Dashboards at `https://$SERVICE_HOST:$CADDY_PORT/grafana`, with the [Loki](../loki/README.md) data source from `provisioning/`. Log in with Keycloak (group `admins` → Admin) or the built-in admin `ADMIN_USER`.

Reset the built-in admin's password after changing `.env`:

```bash
services/ctl.sh exec grafana grafana cli admin reset-admin-password "$ADMIN_PASSWORD"
```

Check it's up:

```bash
curl -k https://$SERVICE_HOST:$CADDY_PORT/grafana/api/health
```

## Notes

- `ctl.sh up` sets the built-in admin's email to the SSO admin's, so the Keycloak login maps onto it.
