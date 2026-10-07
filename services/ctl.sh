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

compose=(docker compose --project-name flywheel "$@")

# `up` needs the state directories to exist with the right owner before the
# bind mounts are created, otherwise dockerd makes them as root.
if [[ $1 == up ]]; then
  for d in caddy/data caddy/config versitygw/buckets versitygw/meta versitygw/iam mlflow prometheus loki grafana; do
    mkdir -p "$STATE_DIR/$d"
  done
  # No --build: compose builds a missing image anyway, and with --build it
  # recreates the container even when the rebuilt image is identical.
  compose=(docker compose --project-name flywheel up -d "${@:2}")
fi

exec "${compose[@]}"
