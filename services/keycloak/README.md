# keycloak

Identity provider (`/auth`, realm `flywheel`) plus oauth2-proxy (`/oauth2`) for apps without a login of their own.

| Service | Login |
|---|---|
| Gitea, Grafana, MLflow | native OIDC ("Sign in with Keycloak"); accounts appear on first login; group `admins` = admin |
| FiftyOne, Rerun, Loki, S3 web UI | oauth2-proxy session cookie for `.$SERVICE_HOST`; groups `admins` and `users` admitted |
| MLflow API | personal access token as password ([mlflow/](../mlflow/README.md)) |
| S3 API | access keys (SigV4), no SSO possible ([versitygw/](../versitygw/README.md)) |

1. Add a person (Keycloak user in group `users`, MLflow token, S3 keys): [Users](../README.md#users).
2. Manage the realm in the admin console (login `ADMIN_USER`): `https://$SERVICE_HOST:$CADDY_PORT/auth/admin/flywheel/console/`.
3. End an oauth2-proxy session: `https://$SERVICE_HOST:$CADDY_PORT/oauth2/sign_out`.
4. After changing `ADMIN_PASSWORD` in `.env`, set it in Keycloak and Grafana.
   ```bash
   services/ctl.sh exec keycloak /opt/keycloak/bin/kcadm.sh set-password -r flywheel --username admin --new-password "$ADMIN_PASSWORD"
   services/ctl.sh exec grafana grafana cli admin reset-admin-password "$ADMIN_PASSWORD"
   ```

**Notes**
- The realm (clients `gitea`, `grafana`, `mlflow`, `oauth2-proxy`, groups, mappers, the admin user) is `realm.json.tmpl`, rendered with the `KC_*` secrets from `.env` and imported on Keycloak's first start only. `ctl.sh up` adds clients missing from an existing realm.
