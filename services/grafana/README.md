# grafana

Dashboards at `https://$SERVICE_HOST:$CADDY_PORT/grafana`, with the [Loki](../loki/README.md) data source from `provisioning/`. Log in with Keycloak (group `admins` → Admin) or the built-in admin `ADMIN_USER`.

After changing `ADMIN_PASSWORD` in `.env`, reset the built-in admin's password:

```bash
services/ctl.sh exec grafana grafana cli admin reset-admin-password "$ADMIN_PASSWORD"
```

To check Grafana is up:

```bash
curl -k https://$SERVICE_HOST:$CADDY_PORT/grafana/api/health
```

## Notes

- Log in as the admin through Keycloak or with `ADMIN_USER`: `ctl.sh up` gives the built-in admin the SSO admin's email, so both land on the same account.
