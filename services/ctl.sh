#!/usr/bin/env bash
# Run `docker compose` for the Store stack on the service node.
#
#   services/ctl.sh up        start (or update) the stack
#   services/ctl.sh down      stop it
#   services/ctl.sh ps|logs|pull|config|<any compose args>
#
# From the login node the command is wrapped in `srun` so it executes on
# $SERVICE_NODE. Containers are owned by that node's dockerd with
# restart=unless-stopped, so they keep running after the srun returns.
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

compose=(docker compose --project-name flywheel "$@")

# Caddy's basic_auth wants a bcrypt hash, never the plaintext password.
if [[ -z ${ADMIN_PASSWORD_HASH:-} ]]; then
  ADMIN_PASSWORD_HASH=$(docker run --rm caddy:2.11 caddy hash-password --plaintext "$ADMIN_PASSWORD")
  export ADMIN_PASSWORD_HASH
fi

# `up` needs the state directories to exist with the right owner before the
# bind mounts are created, otherwise dockerd makes them as root.
if [[ $1 == up ]]; then
  for d in caddy/data caddy/config versitygw/buckets versitygw/meta versitygw/iam mlflow prometheus loki grafana; do
    mkdir -p "$STATE_DIR/$d"
  done
  compose=(docker compose --project-name flywheel up -d --build "${@:2}")
fi

if [[ $(hostname -s) == "${SERVICE_NODE%%.*}" ]]; then
  exec "${compose[@]}"
fi

pty=(); [[ -t 0 ]] && pty=(--pty)   # interactive `logs -f`, `exec` need a tty; scripts don't
exec srun --partition="$SERVICE_PARTITION" --nodelist="$SERVICE_NODE" \
  --job-name=flywheel-services --time=00:30:00 --cpus-per-task=2 --mem=4G \
  --chdir="$here" "${pty[@]}" "${compose[@]}"
