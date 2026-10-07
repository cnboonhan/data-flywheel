#!/usr/bin/env bash
# Run `docker compose` for the Store stack on the service node.
#
#   services/ctl.sh up        start (or update) the stack
#   services/ctl.sh down      stop it
#   services/ctl.sh ps|logs|pull|config|<any compose args>
#
# When run elsewhere (e.g. the login node) it re-runs itself on $SERVICE_NODE
# over ssh; the repo and the state dir are on /tier1, so paths are the same.
# Containers are owned by that node's dockerd with restart=unless-stopped, so
# they keep running after this script returns.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$here"

if [[ ! -f .env ]]; then
  echo "services/.env is missing; copy .env.example and fill in the secrets." >&2
  exit 1
fi
set -a; source .env; set +a

if [[ $# -eq 0 ]]; then
  sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

if [[ $(hostname -s) != "${SERVICE_NODE%%.*}" ]]; then
  # A login shell, so the node's PATH (docker lives under /cm) is set up.
  tty=(); [[ -t 0 ]] && tty=(-t)   # interactive `logs -f`, `exec` need a tty
  exec ssh "${tty[@]}" "$SERVICE_NODE" "bash -lc $(printf '%q' "cd $(printf '%q' "$here") && ./ctl.sh $(printf '%q ' "$@")")"
fi

# Caddy's basic_auth wants a bcrypt hash, never the plaintext password. bcrypt
# salts differ on every run, so cache the hash (keyed by the password's sha256)
# or Caddy's environment would change, and the container be recreated, on
# every `up`.
if [[ -z ${ADMIN_PASSWORD_HASH:-} ]]; then
  mkdir -p "$STATE_DIR/caddy"
  cache="$STATE_DIR/caddy/admin.hash"
  key=$(printf '%s' "$ADMIN_PASSWORD" | sha256sum | cut -d' ' -f1)
  if [[ ! -f $cache || $(cut -d' ' -f1 "$cache") != "$key" ]]; then
    hash=$(docker run --rm caddy:2.11 caddy hash-password --plaintext "$ADMIN_PASSWORD")
    (umask 077; printf '%s %s\n' "$key" "$hash" > "$cache")
  fi
  ADMIN_PASSWORD_HASH=$(cut -d' ' -f2 "$cache")
  export ADMIN_PASSWORD_HASH
fi

compose=(docker compose --project-name flywheel)

gitea() { "${compose[@]}" exec -T gitea gitea --config /etc/gitea/app.ini "$@"; }

# First-run setup Gitea can't take from env: the admin user, the runner's
# registration token, and the `pipelines` repo with the example workflow.
bootstrap_gitea() {
  for _ in $(seq 30); do
    "${compose[@]}" exec -T gitea curl -fs http://localhost:3000/gitea/api/healthz >/dev/null 2>&1 && break
    sleep 2
  done

  if ! gitea admin user list --admin 2>/dev/null | awk 'NR>1 {print $2}' | grep -qx "$ADMIN_USER"; then
    gitea admin user create --admin --username "$ADMIN_USER" --password "$ADMIN_PASSWORD" \
      --email "$ADMIN_USER@flywheel.local" --must-change-password=false
  fi

  if [[ ! -s $STATE_DIR/act_runner/.runner && ! -s $STATE_DIR/act_runner/token ]]; then
    (umask 077; gitea actions generate-runner-token > "$STATE_DIR/act_runner/token")
  fi

  local api="http://localhost:3000/gitea/api/v1" auth="$ADMIN_USER:$ADMIN_PASSWORD"
  if ! "${compose[@]}" exec -T gitea curl -fs -u "$auth" "$api/repos/$ADMIN_USER/pipelines" >/dev/null 2>&1; then
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -H 'Content-Type: application/json' \
      -d '{"name":"pipelines","private":true,"auto_init":true,"default_branch":"main","description":"Dataset processing and mixing pipelines (Gitea Actions)"}' \
      "$api/user/repos" >/dev/null
    for s in S3_ACCESS_KEY:$ADMIN_USER S3_SECRET_KEY:$ADMIN_PASSWORD; do
      "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
        -d "{\"data\":\"${s#*:}\"}" "$api/repos/$ADMIN_USER/pipelines/actions/secrets/${s%%:*}" >/dev/null
    done
  fi
  # Seed files go in once; edit them in the pipelines repo afterwards.
  local f path
  for f in gitea/examples/*.yml fiftyone/*.py xpolicylab/*.py slurm/*.sbatch; do
    case $f in
      gitea/examples/*) path=".gitea/workflows/$(basename "$f")" ;;
      *) path="$f" ;;
    esac
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" "$api/repos/$ADMIN_USER/pipelines/contents/$path" >/dev/null 2>&1 && continue
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -H 'Content-Type: application/json' \
      -d "{\"content\":\"$(base64 -w0 "$f")\",\"message\":\"Add $path\"}" \
      "$api/repos/$ADMIN_USER/pipelines/contents/$path" >/dev/null
  done
}

if [[ $1 == up ]]; then
  # The state directories must exist with the right owner before the bind
  # mounts are created, otherwise dockerd makes them as root.
  for d in caddy/data caddy/config versitygw/buckets versitygw/meta versitygw/iam mlflow prometheus loki grafana gitea/data gitea/config act_runner mongo fiftyone; do
    mkdir -p "$STATE_DIR/$d"
  done
  # Gitea first so the runner finds its token when it starts. No --build:
  # compose builds a missing image anyway, and with --build it recreates the
  # container even when the rebuilt image is identical.
  "${compose[@]}" up -d gitea
  bootstrap_gitea
  exec "${compose[@]}" up -d "${@:2}"
fi

exec "${compose[@]}" "$@"
