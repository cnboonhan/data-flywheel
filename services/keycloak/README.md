# keycloak

Identity provider (`/auth`, realm `flywheel`) plus oauth2-proxy (`/oauth2`) for apps without a login of their own.

| Service | Login |
|---|---|
| Gitea, Grafana, MLflow | native OIDC ("Sign in with Keycloak"); accounts appear on first login; group `admins` = admin |
| FiftyOne, Rerun, Loki, S3 web UI | oauth2-proxy session cookie for `.$SERVICE_HOST`; groups `admins` and `users` admitted |
| MLflow API | personal access token as password ([mlflow/](../mlflow/README.md)) |
| S3 API | access keys (SigV4), no SSO possible ([versitygw/](../versitygw/README.md)) |

1. To add a person (Keycloak user in group `users`, MLflow token, S3 keys), follow [Users](../README.md#users).
2. To manage users, groups and clients, open the admin console at `https://$SERVICE_HOST:$CADDY_PORT/auth/admin/flywheel/console/` and log in as `ADMIN_USER`.
3. To sign out of FiftyOne, Rerun and the other proxied apps, open `https://$SERVICE_HOST:$CADDY_PORT/oauth2/sign_out`.
4. After changing `ADMIN_PASSWORD` in `.env`, set the new password in Keycloak and Grafana.
   ```bash
   services/ctl.sh exec keycloak /opt/keycloak/bin/kcadm.sh set-password -r flywheel --username admin --new-password "$ADMIN_PASSWORD"
   services/ctl.sh exec grafana grafana cli admin reset-admin-password "$ADMIN_PASSWORD"
   ```

**Notes**
- Change the realm (clients `gitea`, `grafana`, `mlflow`, `oauth2-proxy`, groups, mappers, the admin user) in `realm.json.tmpl`. Keycloak imports it, filled in with the `KC_*` secrets from `.env`, on its first start only; afterwards `ctl.sh up` adds missing clients, and any other change has to be made in the admin console.
