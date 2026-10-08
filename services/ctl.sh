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

  # Keycloak as an OIDC login source ("Sign in with Keycloak"; accounts auto-register).
  # Gitea validates the discovery URL when adding it, so wait until Keycloak
  # answers through Caddy (its realm import takes a while on first start).
  local discovery="https://$SERVICE_HOST:$CADDY_PORT/auth/realms/flywheel/.well-known/openid-configuration"
  if ! gitea admin auth list 2>/dev/null | grep -q 'keycloak'; then
    for _ in $(seq 60); do
      "${compose[@]}" exec -T gitea curl -fs --cacert /ca/root.crt "$discovery" >/dev/null 2>&1 && break
      sleep 5
    done
    gitea admin auth add-oauth --name keycloak --provider openidConnect --key gitea --secret "$KC_GITEA_SECRET" \
      --auto-discover-url "$discovery" \
      --scopes openid --scopes profile --scopes email --group-claim-name groups --skip-local-2fa
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
  # Repo variables the workflows read (secrets hold credentials; these are plain URLs).
  for v in "RERUN_BASE=https://rerun.$SERVICE_HOST:$CADDY_PORT" "SERVICE_URL=https://$SERVICE_HOST:$CADDY_PORT"; do
    body=$(python3 -c 'import json,sys; print(json.dumps({"value": sys.argv[1]}))' "${v#*=}")
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' -d "$body" "$api/repos/$ADMIN_USER/pipelines/actions/variables/${v%%=*}" >/dev/null 2>&1 \
      || "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X POST -H 'Content-Type: application/json' -d "$body" "$api/repos/$ADMIN_USER/pipelines/actions/variables/${v%%=*}" >/dev/null
  done
  for f in gitea/examples/*.yml fiftyone/*.py xpolicylab/*.py slurm/*.sbatch slurm/follow.sh \
           slurm/conda-shim/bin/conda slurm/conda-shim/etc/profile.d/conda.sh slurm/robodojo-shim/sitecustomize.py; do
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

# Let Actions jobs submit Slurm work: an SSH key for the runner, locked to the
# slurm-submit forced command on SLURM_LOGIN_HOST, stored as repo secrets; and
# the credentials Slurm jobs need, in a file only the submitting user reads.
bootstrap_slurm() {
  local dir="$STATE_DIR/act_runner/ssh" key api="http://localhost:3000/gitea/api/v1" auth="$ADMIN_USER:$ADMIN_PASSWORD"
  mkdir -p "$dir" "$STATE_DIR/slurm-logs"
  # The checkout of the pipelines repo that slurm-submit updates and the jobs read (PIPELINES_ROOT).
  local ca="$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt" repo="$STATE_DIR/pipelines"
  (umask 077; printf 'https://%s:%s@%s:%s\n' "$ADMIN_USER" "$ADMIN_PASSWORD" "$SERVICE_HOST" "$CADDY_PORT" > "$STATE_DIR/.pipelines-credentials")
  if [[ ! -d $repo/.git ]]; then
    git -c "http.sslCAInfo=$ca" -c "credential.helper=store --file=$STATE_DIR/.pipelines-credentials" \
      clone -q "https://$SERVICE_HOST:$CADDY_PORT/gitea/$ADMIN_USER/pipelines.git" "$repo"
  fi
  git -C "$repo" config http.sslCAInfo "$ca"
  git -C "$repo" config credential.helper "store --file=$STATE_DIR/.pipelines-credentials"
  [[ -f $dir/id_ed25519 ]] || ssh-keygen -q -t ed25519 -N "" -C "flywheel-actions-runner" -f "$dir/id_ed25519"
  key=$(cut -d' ' -f1,2 "$dir/id_ed25519.pub")
  local line="command=\"$here/slurm/slurm-submit\",restrict $key flywheel-actions-runner"
  mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  # One forced-command line for the runner; drop stale ones (an earlier key, or the repo at another path).
  grep -v 'flywheel-actions-runner' ~/.ssh/authorized_keys > ~/.ssh/authorized_keys.tmp || true
  echo "$line" >> ~/.ssh/authorized_keys.tmp && mv ~/.ssh/authorized_keys.tmp ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  local mltok; mltok=$(mlflow_token "$ADMIN_USER" "$ADMIN_PASSWORD") || { echo "warning: no MLflow token for jobs" >&2; mltok=; }
  (umask 077; cat > "$STATE_DIR/slurm.env" <<EOF
MLFLOW_TRACKING_URI=https://$SERVICE_HOST:$CADDY_PORT/mlflow
MLFLOW_TRACKING_USERNAME=$ADMIN_USER
MLFLOW_TRACKING_PASSWORD=$mltok
MLFLOW_TRACKING_SERVER_CERT_PATH=$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt
MLFLOW_DISABLE_AGENT_HINT=1
S3_ENDPOINT_URL=https://s3.$SERVICE_HOST:$CADDY_PORT
AWS_CA_BUNDLE=$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt
AWS_ACCESS_KEY_ID=$ADMIN_USER
AWS_SECRET_ACCESS_KEY=$ADMIN_PASSWORD
AWS_DEFAULT_REGION=${S3_REGION:-us-east-1}
UV_CACHE_DIR=$STATE_DIR/uv-cache
# Where this checkout and the shared state live, for the Slurm scripts (their defaults assume /tier1).
FLYWHEEL_ROOT=$(cd "$here/.." && pwd)
PROJECT_ROOT=$(cd "$here/.." && pwd)/eval/system1/RoboDojo
ENVS_DIR=$STATE_DIR/envs
ROBODOJO_DIR=$STATE_DIR/robodojo
BUCKETS_DIR=$STATE_DIR/versitygw/buckets
EOF
  )
  for s in "SLURM_SSH_KEY=$(cat "$dir/id_ed25519")" "SLURM_SSH_HOST=$USER@$SLURM_LOGIN_HOST"; do
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"data": sys.argv[1]}))' "${s#*=}")" \
      "$api/repos/$ADMIN_USER/pipelines/actions/secrets/${s%%=*}" >/dev/null
  done
}

# Keycloak's realm import only seeds a new realm; clients added to the template later
# (mlflow) are created here from the rendered realm file when missing.
bootstrap_keycloak() {
  local kc=(docker compose --project-name flywheel exec -T keycloak /opt/keycloak/bin/kcadm.sh) realm="$STATE_DIR/keycloak/import/flywheel-realm.json" c
  for _ in $(seq 1 90); do
    "${kc[@]}" config credentials --server http://localhost:8080/auth --realm master --user "$ADMIN_USER" --password "$ADMIN_PASSWORD" >/dev/null 2>&1 && break
    sleep 2
  done
  for c in $(python3 -c 'import json,sys; print(" ".join(c["clientId"] for c in json.load(open(sys.argv[1]))["clients"]))' "$realm"); do
    "${kc[@]}" get clients -r flywheel -q "clientId=$c" --fields clientId 2>/dev/null | grep -q "\"$c\"" && continue
    python3 -c 'import json,sys; print(json.dumps(next(x for x in json.load(open(sys.argv[1]))["clients"] if x["clientId"] == sys.argv[2])))' "$realm" "$c" \
      | "${kc[@]}" create clients -r flywheel -f - >/dev/null && echo "keycloak: created client $c"
  done
}

# MLflow (mlflow-oidc-auth): wait until the server answers. Inside the compose network MLflow serves at the
# root; the /mlflow prefix only exists through Caddy (X-Forwarded-Prefix).
bootstrap_mlflow() {
  for _ in $(seq 1 60); do
    "${compose[@]}" exec -T gitea curl -fs -H "Host: $SERVICE_HOST" http://mlflow:5000/health >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "warning: mlflow is not answering; access tokens can't be minted" >&2
}

# MLflow access token for API clients (MLFLOW_TRACKING_USERNAME=<name>, MLFLOW_TRACKING_PASSWORD=<token>):
# sign the user in with a Keycloak password grant, then (re)issue their "default" token, valid a year.
mlflow_token() {
  local name=$1 pw=$2 tok
  tok=$("${compose[@]}" exec -T gitea curl -fs -d grant_type=password -d client_id=mlflow -d "client_secret=$KC_MLFLOW_SECRET" \
        -d "username=$name" -d "password=$pw" -d scope=openid http://keycloak:8080/auth/realms/flywheel/protocol/openid-connect/token \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])') || return 1
  "${compose[@]}" exec -T gitea curl -fs -X PATCH -H "Authorization: Bearer $tok" -H "Host: $SERVICE_HOST" -H 'Content-Type: application/json' \
    -d "{\"expiration\":\"$(date -u -d '+365 days' +%Y-%m-%dT%H:%M:%SZ)\"}" http://mlflow:5000/api/2.0/mlflow/users/access-token \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
}

# Add a person to every service under one username:
#   ctl.sh user add <name> <email> [password]
# Keycloak (SSO: Gitea, Grafana, MLflow, FiftyOne, Rerun, Prometheus, Loki), an MLflow
# access token for API clients (printed once) and the S3 gateway (an access key
# pair, printed once). Password defaults to a random one, printed.
# Grafana creates its own local admin (email admin@localhost). Give it the SSO admin's
# email so the Keycloak login maps onto it instead of trying to create a second "admin".
bootstrap_grafana() {
  local api="http://grafana:3000/grafana/api" auth="$ADMIN_USER:$ADMIN_PASSWORD"
  for _ in $(seq 1 60); do
    "${compose[@]}" exec -T gitea curl -fs "$api/health" >/dev/null 2>&1 && break
    sleep 2
  done
  # Once the admin has signed in through Keycloak, Grafana marks the user external and refuses edits; by then the email already matches.
  "${compose[@]}" exec -T gitea curl -fs -u "$auth" "$api/users/1" | grep -q "\"email\":\"$ADMIN_USER@flywheel.local\"" && return 0
  "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
    -d "{\"login\":\"$ADMIN_USER\",\"email\":\"$ADMIN_USER@flywheel.local\",\"name\":\"$ADMIN_USER\"}" "$api/users/1" >/dev/null \
    || echo "warning: could not set the Grafana admin email (SSO login as $ADMIN_USER may fail)" >&2
}

user_add() {
  local name=${1:?name} email=${2:?email} pw=${3:-$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)}
  local kc=(docker compose --project-name flywheel exec -T keycloak /opt/keycloak/bin/kcadm.sh)
  "${kc[@]}" config credentials --server http://localhost:8080/auth --realm master --user "$ADMIN_USER" --password "$ADMIN_PASSWORD" >/dev/null
  if "${kc[@]}" get users -r flywheel -q "username=$name" --fields username 2>/dev/null | grep -q "\"$name\""; then
    echo "keycloak: $name exists"
  else
    "${kc[@]}" create users -r flywheel -s "username=$name" -s "email=$email" -s enabled=true -s emailVerified=true >/dev/null
    "${kc[@]}" set-password -r flywheel --username "$name" --new-password "$pw"
    local gid; gid=$("${kc[@]}" get groups -r flywheel -q search=users --fields id,name 2>/dev/null | python3 -c 'import sys,json; print([g["id"] for g in json.load(sys.stdin) if g["name"]=="users"][0])')
    local uid; uid=$("${kc[@]}" get users -r flywheel -q "username=$name" --fields id | python3 -c 'import sys,json; print(json.load(sys.stdin)[0]["id"])')
    "${kc[@]}" update "users/$uid/groups/$gid" -r flywheel -n >/dev/null
    echo "keycloak: created $name (group users)"
  fi

  local tok
  if tok=$(mlflow_token "$name" "$pw" 2>/dev/null) && [[ -n $tok ]]; then
    echo "mlflow: access token $tok   (MLFLOW_TRACKING_USERNAME=$name MLFLOW_TRACKING_PASSWORD=<token>; shown once)"
  else
    echo "mlflow: no token minted (sign in at /mlflow with Keycloak; Profile > Tokens issues one)"
  fi

  local secret; secret=$(openssl rand -hex 20)
  if docker compose --project-name flywheel exec -T -e ROOT_ACCESS_KEY_ID="$ADMIN_USER" -e ROOT_SECRET_ACCESS_KEY="$ADMIN_PASSWORD" versitygw \
       versitygw admin -a "$ADMIN_USER" -s "$ADMIN_PASSWORD" -er http://localhost:7070 create-user -a "$name" -s "$secret" -r user >/dev/null 2>&1; then
    echo "s3: access key $name, secret $secret   (endpoint https://s3.$SERVICE_HOST:$CADDY_PORT; shown once)"
  else
    echo "s3: $name exists (or create failed); secret unchanged"
  fi
  echo "password: $pw   (Keycloak: every web UI, including MLflow)"
}

if [[ $1 == user ]]; then
  case ${2:-} in
    add) user_add "${@:3}" ;;
    *) echo "usage: ctl.sh user add <name> <email> [password]" >&2; exit 2 ;;
  esac
  exit
fi

if [[ $1 == up ]]; then
  # The state directories must exist with the right owner before the bind
  # mounts are created, otherwise dockerd makes them as root.
  for d in caddy/data caddy/config versitygw/buckets versitygw/meta versitygw/iam mlflow prometheus loki grafana gitea/data gitea/config act_runner mongo fiftyone keycloak/db keycloak/import versitygw/buckets/processed/rerun; do
    mkdir -p "$STATE_DIR/$d"
  done
  # Keycloak imports the realm (clients, groups, the admin user) on first start.
  (umask 077; python3 -c '
import os, re, sys
t = open("keycloak/realm.json.tmpl").read()
print(re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), t))' > "$STATE_DIR/keycloak/import/flywheel-realm.json")
  # Everything but the runner first: the bootstrap needs Gitea, Caddy and
  # Keycloak up, and the runner needs the token the bootstrap writes. No
  # --build: compose builds a missing image anyway, and with --build it
  # recreates the container even when the rebuilt image is identical.
  # Caddy first: Gitea and oauth2-proxy bind-mount its CA certificate, and if that file doesn't exist yet
  # (first start) dockerd creates a directory in its place, which then stops Caddy from writing the CA.
  ca="$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt"
  mkdir -p "$(dirname "$ca")"   # as us; a bind mount onto a missing path would make dockerd create the parents as root
  if [[ ! -f $ca ]]; then
    "${compose[@]}" up -d --no-deps caddy   # --no-deps: its dependents are the ones mounting the CA
    for _ in $(seq 1 60); do [[ -f $ca ]] && break; sleep 1; done
    [[ -f $ca ]] || { echo "caddy did not create its CA at $ca" >&2; exit 1; }
  fi
  "${compose[@]}" up -d "${@:2}" $("${compose[@]}" config --services | grep -vx act_runner)
  bootstrap_gitea
  bootstrap_grafana
  bootstrap_keycloak
  bootstrap_mlflow
  [[ -n ${SLURM_LOGIN_HOST:-} ]] && bootstrap_slurm
  exec "${compose[@]}" up -d "${@:2}"
fi

exec "${compose[@]}" "$@"
