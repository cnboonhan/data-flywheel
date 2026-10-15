#!/usr/bin/env bash
# Run `docker compose` for the Store stack on the service node.
#
#   services/ctl.sh up        start (or update) the stack
#   services/ctl.sh down      stop it
#   services/ctl.sh setup     (re)build the Slurm-side environments (Gitea workflow setup-envs)
#   services/ctl.sh s3-policies  re-apply the S3 bucket policies (after editing versitygw/upload-only.txt)
#   services/ctl.sh ps|logs|pull|config|<any compose args>
#   services/ctl.sh triton <compose args>   Triton's own project on TRITON_NODE (ps, logs, build, ...)
#   services/ctl.sh dynamo render|up|down|<compose args>   the LLMs on DYNAMO_HEAD + DYNAMO_WORKER (dynamo/README.md)
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
# Triton runs on TRITON_NODE (triton/compose.yml); containers here reach it by address, as they can't resolve node names.
TRITON_ADDR=$(getent hosts "$TRITON_NODE" | awk '{print $1; exit}'); export TRITON_ADDR
LLM_GATEWAY_ADDR=$(getent hosts "${DYNAMO_HEAD:-}" | awk '{print $1; exit}' || true); export LLM_GATEWAY_ADDR   # dynamo/

if [[ $# -eq 0 ]]; then
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'   # lines 2-10: the usage block at the top of this file
  exit 1
fi

if [[ $(hostname -s) != "${SERVICE_NODE%%.*}" ]]; then
  # A login shell, so the node's PATH (docker lives under /cm) is set up.
  tty=(); [[ -t 0 ]] && tty=(-t)   # interactive `logs -f`, `exec` need a tty
  exec ssh "${tty[@]}" "$SERVICE_NODE" "bash -lc $(printf '%q' "cd $(printf '%q' "$here") && ./ctl.sh $(printf '%q ' "$@")")"
fi

compose=(docker compose --project-name flywheel)

# On a node whose GPUs' memory is NUMA nodes of its own (GB300), Linux puts file cache there too, and the GPUs fill up
# without any GPU process. The stack's containers use no GPU here: hold their memory, file cache included, to the nodes
# with CPUs. Running containers are updated in place (`docker update`), so nothing restarts. Never for GPU containers:
# their CUDA memory is on exactly those nodes. Job containers get the same from act_runner/config.yaml.
keep_off_gpu_memory() {
  local d mems=() all=()
  for d in /sys/devices/system/node/node[0-9]*; do
    all+=("${d##*node}"); [[ -n $(cat "$d/cpulist") ]] && mems+=("${d##*node}")
  done
  (( ${#mems[@]} < ${#all[@]} )) || return 0   # every NUMA node has CPUs: nothing to keep off
  docker update --cpuset-mems "$(IFS=,; echo "${mems[*]}")" $("${compose[@]}" ps -q) >/dev/null
}

# Dynamo (dynamo/README.md): serves the models whose MLflow alias `dynamo` is set, where models.yaml places them (the
# trays, or a node's GPUs). render: write each node's compose file from models.yaml and the aliases (dynamo/render.py).
# up: render, remove the engines no longer served (they hold GPUs), take down nodes left with nothing, bring every
# node's services up (unchanged ones keep running) and wait for each served model. sync: up, only when the aliases moved
# or an engine is down; timer: run sync every minute on DYNAMO_HEAD, so setting or moving an alias is all a change takes.
dynamo() {
  local out="$STATE_DIR/dynamo"
  on() { local node=$1; shift
    ssh "$node" "bash -lc $(printf '%q' "cd $(printf '%q' "$out") && docker compose -p flywheel-dynamo --env-file $(printf '%q' "$here/.env") -f $node.json $(printf '%q ' "$@")")"; }
  q() { python3 -c "import json,sys; d=json.load(open('$out/nodes.json')); print($1)" 2>/dev/null; }   # from nodes.json
  engines() { ssh "$1" "bash -lc 'docker ps $2 --filter name=flywheel-dynamo-engine- --format {{.Names}}'" | LC_ALL=C sort | xargs; }
  wanted() { python3 -c 'import json,sys; print(" ".join(sorted("flywheel-dynamo-"+s+"-1" for s in json.load(open(sys.argv[1]))["services"] if s.startswith("engine-"))))' "$out/$1.json" | tr ' ' '\n' | LC_ALL=C sort | xargs; }
  case ${1:-} in
    render)
      mkdir -p "$out/cache"
      (set -a; . "$STATE_DIR/flywheel.env"; uv run -q --with pyyaml==6.0.2 python3 "$here/dynamo/render.py" "$out") || return 1
      for m in $(q '" ".join(d["served"])'); do mkdir -p "$out/cache/$m"; done
      for node in $(q '" ".join(d["nodes"])'); do mkdir -p "$out/etcd-$node"; done ;;   # else dockerd makes them, root-owned
    up)
      exec 9>"$out/up.lock"   # one up at a time, by hand or from the timer's sync
      flock -n 9 || { echo "dynamo: another up is running" >&2; return 1; }
      local before node m sv old
      before=$(q '" ".join(d["nodes"])')
      dynamo render || return 1
      rm -f "$out/failed"
      for node in $before; do   # nodes that no longer run anything
        [[ " $(q '" ".join(d["nodes"])') " == *" $node "* ]] ||
          { echo "dynamo: nothing left on $node"; ssh "$node" "bash -lc 'docker compose -p flywheel-dynamo down --remove-orphans'"; }
      done
      for node in $(q '" ".join(d["nodes"])'); do   # engines no longer served hold the GPUs: remove them first
        old=$(LC_ALL=C comm -23 <(engines "$node" -a | tr ' ' '\n' | sed '/^$/d' | LC_ALL=C sort) \
                                <(wanted "$node" | tr ' ' '\n' | sed '/^$/d' | LC_ALL=C sort) | xargs)
        [[ -z $old ]] || { echo "dynamo: removing $old on $node"; ssh "$node" "bash -lc 'docker rm -f $old'" >/dev/null; }
      done
      for node in $(q '" ".join(sorted(d["nodes"], key=lambda n: n == "'"$DYNAMO_HEAD"'"))'); do   # the head (rank 0) last
        # linger: else logind deletes the engines' /dev/shm files whenever our last ssh session to the node ends
        ssh "$node" loginctl enable-linger
        on "$node" up -d --build --remove-orphans
      done
      for m in $(q '" ".join(d["served"])'); do
        sv=($(q "' '.join(map(str, (d['served']['$m']['node'], d['served']['$m']['system'], d['served']['$m']['tray'])))"))
        echo "dynamo: $m loading on ${sv[0]} (minutes: weights from /tier1, then compilation)"
        until ssh "${sv[0]}" "curl -fs http://127.0.0.1:${sv[1]}/health" >/dev/null 2>&1; do
          for node in "${sv[0]}" $([[ ${sv[2]} == True ]] && echo "$DYNAMO_WORKER"); do
            [[ $(ssh "$node" "bash -lc 'docker inspect -f {{.State.Running}} flywheel-dynamo-engine-$m-1'") == true ]] ||
              { echo "dynamo: engine-$m exited on $node (ctl.sh dynamo on $node logs engine-$m)" >&2; cksum "$out"/*.json > "$out/failed"; return 1; }
            # a worker that dies leaves its container up and the other rank waiting forever
            ! ssh "$node" "bash -lc 'docker logs flywheel-dynamo-engine-$m-1 2>&1 | grep -q -E \"WorkerProc hit an exception|EngineCore encountered a fatal error\"'" ||
              { echo "dynamo: engine-$m failed on $node (ctl.sh dynamo on $node logs engine-$m)" >&2; cksum "$out"/*.json > "$out/failed"; return 1; }
          done
          sleep 10
        done
        echo "dynamo: $m serving on ${sv[0]}"
      done ;;
    sync)
      local before node
      before=$(cat "$out"/*.json 2>/dev/null | cksum)
      dynamo render >/dev/null || return 1
      if [[ $(cat "$out"/*.json | cksum) == "$before" ]]; then   # same files: only check the engines are up
        local down=
        for node in $(q '" ".join(d["nodes"])'); do [[ $(engines "$node") == "$(wanted "$node")" ]] || down=1; done
        [[ -z $down ]] && return 0
        # a start that failed isn't retried until the aliases or the config change (ctl.sh dynamo up retries by hand)
        [[ -f $out/failed ]] && cmp -s "$out/failed" <(cksum "$out"/*.json) && return 0
      fi
      flock -n "$out/up.lock" true || return 0   # an up is running already (by hand or the last sync)
      echo "dynamo: serving $(q '", ".join(d["served"])')"
      dynamo up ;;
    timer)   # install or refresh the sync timer on DYNAMO_HEAD (linger keeps user timers running without a login)
      ssh "$DYNAMO_HEAD" "mkdir -p ~/.config/systemd/user && cat > ~/.config/systemd/user/flywheel-dynamo-sync.service && cat > ~/.config/systemd/user/flywheel-dynamo-sync.timer <<'TIMER'
[Unit]
Description=Serve the Dynamo models whose MLflow alias dynamo is set

[Timer]
OnBootSec=2min
OnUnitInactiveSec=1min

[Install]
WantedBy=timers.target
TIMER
loginctl enable-linger && systemctl --user daemon-reload && systemctl --user enable --now flywheel-dynamo-sync.timer && systemctl --user list-timers flywheel-dynamo-sync.timer --no-pager" <<UNIT
[Unit]
Description=Serve the Dynamo models whose MLflow alias dynamo is set

[Service]
Type=oneshot
ExecStart=/bin/bash -lc 'flock -n -E 0 $out/sync.lock $here/ctl.sh dynamo sync'
UNIT
      ;;
    down)
      for node in $(q '" ".join(d["nodes"])'); do on "$node" down --remove-orphans; done ;;
    on)      # compose on one node: ctl.sh dynamo on <node> ps|logs|...
      on "${@:2}" ;;
    worker)
      on "$DYNAMO_WORKER" "${@:2}" ;;
    *)
      on "$DYNAMO_HEAD" "$@" ;;
  esac
}

# docker compose for Triton's own project on TRITON_NODE (over ssh; the checkout and STATE_DIR are on shared storage).
triton_compose() {
  ssh "$TRITON_NODE" "bash -lc $(printf '%q' "cd $(printf '%q' "$here/triton") && docker compose --project-name flywheel-triton --env-file ../.env -f compose.yml $(printf '%q ' "$@")")"
}

gitea() { "${compose[@]}" exec -T gitea gitea --config /etc/gitea/app.ini "$@"; }

# First-run setup Gitea can't take from env: the admin user, the runner's
# registration token, and the `pipelines` repo with the example workflow.
bootstrap_gitea() {
  for _ in $(seq 30); do   # wait up to 1 min (tries x sleep), untuned
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
    for _ in $(seq 60); do   # up to 5 min, untuned
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
  # Optional Hugging Face token for gated repos (download-datasets-hf, download-models-hf): HF_TOKEN in .env, synced on every up.
  if [[ -n ${HF_TOKEN:-} ]]; then
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"data": sys.argv[1]}))' "$HF_TOKEN")" \
      "$api/repos/$ADMIN_USER/pipelines/actions/secrets/HF_TOKEN" >/dev/null
  fi
  # The header key Triton requires (sync-triton), synced on every up.
  "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"data": sys.argv[1]}))' "$TRITON_KEY")" \
    "$api/repos/$ADMIN_USER/pipelines/actions/secrets/TRITON_KEY" >/dev/null
  # Path token for raw bags in Rerun links (sync-fiftyone-raw): RERUN_RAW_TOKEN in .env, synced on every up.
  if [[ -n ${RERUN_RAW_TOKEN:-} ]]; then
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"data": sys.argv[1]}))' "$RERUN_RAW_TOKEN")" \
      "$api/repos/$ADMIN_USER/pipelines/actions/secrets/RERUN_RAW_TOKEN" >/dev/null
  fi
  # Seed files go in once; edit them in the pipelines repo afterwards.
  local f path
  # Repo variables the workflows read (secrets hold credentials; these are plain URLs).
  for v in "RERUN_BASE=https://rerun.$SERVICE_HOST:$CADDY_PORT" "SERVICE_URL=https://$SERVICE_HOST:$CADDY_PORT" \
           "STATE_DIR=$STATE_DIR" "FLYWHEEL_ROOT=$(cd "$here/.." && pwd)" "USER_LOCAL=$HOME/.local" "RUN_UID=$SERVICE_UID" "RUN_GID=$SERVICE_GID" \
           "TRITON_URL=http://$TRITON_ADDR:8100" "TRITON_PYTHON_GPU=0" "TRITON_VLLM_GPU=1"; do   # Triton's GPUs are 0, 1 inside it
    body=$(python3 -c 'import json,sys; print(json.dumps({"value": sys.argv[1]}))' "${v#*=}")
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' -d "$body" "$api/repos/$ADMIN_USER/pipelines/actions/variables/${v%%=*}" >/dev/null 2>&1 \
      || "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X POST -H 'Content-Type: application/json' -d "$body" "$api/repos/$ADMIN_USER/pipelines/actions/variables/${v%%=*}" >/dev/null
  done
  for f in gitea/{setup,ingest,adapter,clean,mix,serve}/*.yml gitea/adapter/*/*.yml gitea/setup/*.sh gitea/adapter/*/*.py \
           gitea/setup/conda-shim/bin/conda gitea/setup/conda-shim/etc/profile.d/conda.sh; do
    [[ -f $f ]] || continue   # a stage folder without workflows yet leaves its glob unmatched
    case $f in
      gitea/*/*.yml|gitea/*/*/*.yml) path=".gitea/workflows/$(basename "$f")" ;;   # Gitea reads workflows from a flat directory
      gitea/setup/conda-shim/*) path="${f#gitea/}" ;;                # copied into the envs by install-robodojo.sh
      gitea/setup/*.sh) path="setup/$(basename "$f")" ;;              # run by the setup-envs workflow
      gitea/adapter/*/*) path="${f#gitea/}" ;;                        # adapter/<source>/<script> the adapter workflows run
      *) path="$f" ;;
    esac
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" "$api/repos/$ADMIN_USER/pipelines/contents/$path" >/dev/null 2>&1 && continue
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -H 'Content-Type: application/json' \
      -d "{\"content\":\"$(base64 -w0 "$f")\",\"message\":\"Add $path\"}" \
      "$api/repos/$ADMIN_USER/pipelines/contents/$path" >/dev/null
  done
}

# What jobs need besides the services: a checkout of the pipelines repo to edit workflows in, the MLflow
# token, and $STATE_DIR/flywheel.env (credentials and paths, in a file only you can read: the setup-envs workflow,
# Slurm jobs and the sim tools load it).
bootstrap_jobs() {
  local api="http://localhost:3000/gitea/api/v1" auth="$ADMIN_USER:$ADMIN_PASSWORD"
  mkdir -p "$STATE_DIR/slurm-logs"
  local ca="$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt" repo="$STATE_DIR/pipelines"
  (umask 077; printf 'https://%s:%s@%s:%s\n' "$ADMIN_USER" "$ADMIN_PASSWORD" "$SERVICE_HOST" "$CADDY_PORT" > "$STATE_DIR/.pipelines-credentials")
  if [[ ! -d $repo/.git ]]; then
    git -c "http.sslCAInfo=$ca" -c "credential.helper=store --file=$STATE_DIR/.pipelines-credentials" \
      clone -q "https://$SERVICE_HOST:$CADDY_PORT/gitea/$ADMIN_USER/pipelines.git" "$repo"
  fi
  git -C "$repo" config http.sslCAInfo "$ca"
  git -C "$repo" config credential.helper "store --file=$STATE_DIR/.pipelines-credentials"
  # Keep the jobs' token while MLflow accepts it: minting a new one revokes the old, which fails every job running on it
  # (a download-models-hf mid-way). A new one only when it is missing, revoked or expired (they last 365 days).
  local mltok; mltok=$(sed -n 's/^MLFLOW_TRACKING_PASSWORD=//p' "$STATE_DIR/flywheel.env" 2>/dev/null)
  if [[ -z $mltok ]] || ! "${compose[@]}" exec -T gitea curl -fs -o /dev/null -u "$ADMIN_USER:$mltok" -H "Host: $SERVICE_HOST" \
       "http://mlflow:5000/api/2.0/mlflow/experiments/search?max_results=1"; then
    mltok=$(mlflow_token "$ADMIN_USER" "$ADMIN_PASSWORD") || { echo "warning: no MLflow token for jobs" >&2; mltok=; }
  fi
  # The same token for Actions jobs that talk to MLflow (download-models-hf): secret MLFLOW_TOKEN, variable MLFLOW_USERNAME.
  if [[ -n $mltok ]]; then
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
      -d "{\"data\":\"$mltok\"}" "$api/repos/$ADMIN_USER/pipelines/actions/secrets/MLFLOW_TOKEN" >/dev/null
    body="{\"value\":\"$ADMIN_USER\"}"
    "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' -d "$body" "$api/repos/$ADMIN_USER/pipelines/actions/variables/MLFLOW_USERNAME" >/dev/null 2>&1 \
      || "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X POST -H 'Content-Type: application/json' -d "$body" "$api/repos/$ADMIN_USER/pipelines/actions/variables/MLFLOW_USERNAME" >/dev/null
  fi
  (umask 077; cat > "$STATE_DIR/flywheel.env" <<EOF
MLFLOW_TRACKING_URI=https://$SERVICE_HOST:$CADDY_PORT/mlflow
MLFLOW_TRACKING_USERNAME=$ADMIN_USER
MLFLOW_TRACKING_PASSWORD=$mltok
MLFLOW_TRACKING_SERVER_CERT_PATH=$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt
MLFLOW_DISABLE_AGENT_HINT=1
# Large artifacts through the server: its presigned URLs point at http://versitygw:7070, which only resolves inside
# the stack's network (downloads hang otherwise).
MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD=false
MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD=false
S3_ENDPOINT_URL=https://s3.$SERVICE_HOST:$CADDY_PORT
AWS_CA_BUNDLE=$STATE_DIR/caddy/data/caddy/pki/authorities/local/root.crt
AWS_ACCESS_KEY_ID=$ADMIN_USER
AWS_SECRET_ACCESS_KEY=$ADMIN_PASSWORD
AWS_DEFAULT_REGION=${S3_REGION:-us-east-1}
UV_CACHE_DIR=$STATE_DIR/uv-cache
# Where this checkout and the shared state live, for the Slurm jobs and setup-envs (their defaults assume /tier1).
FLYWHEEL_ROOT=$(cd "$here/.." && pwd)
PROJECT_ROOT=$(cd "$here/.." && pwd)/eval/system1/RoboDojo
ENVS_DIR=$STATE_DIR/envs
ROBODOJO_DIR=$STATE_DIR/robodojo
BUCKETS_DIR=$STATE_DIR/versitygw/buckets
DATA_ROOT=$STATE_DIR/xpolicylab
TRITON_URL=triton.$SERVICE_HOST:$CADDY_PORT
# The LLM gateway (Dynamo's models, OpenAI API, no token): sim tools' VLM, scripts
LLM_URL=https://llm.$SERVICE_HOST:$CADDY_PORT/v1
SERVICE_NODE=$SERVICE_NODE
TRITON_TOKEN=${TRITON_TOKEN:-}
EOF
  )
}

# Keycloak's realm import only seeds a new realm; clients added to the template later
# (mlflow) are created here from the rendered realm file when missing.
bootstrap_keycloak() {
  local kc=(docker compose --project-name flywheel exec -T keycloak /opt/keycloak/bin/kcadm.sh) realm="$STATE_DIR/keycloak/import/flywheel-realm.json" c
  for _ in $(seq 1 90); do   # wait up to 3 min for Keycloak, untuned
    "${kc[@]}" config credentials --server http://localhost:8080/auth --realm master --user "$ADMIN_USER" --password "$ADMIN_PASSWORD" >/dev/null 2>&1 && break
    sleep 2
  done
  for c in $(python3 -c 'import json,sys; print(" ".join(c["clientId"] for c in json.load(open(sys.argv[1]))["clients"]))' "$realm"); do
    "${kc[@]}" get clients -r flywheel -q "clientId=$c" --fields clientId 2>/dev/null | grep -q "\"$c\"" && continue
    python3 -c 'import json,sys; print(json.dumps(next(x for x in json.load(open(sys.argv[1]))["clients"] if x["clientId"] == sys.argv[2])))' "$realm" "$c" \
      | "${kc[@]}" create clients -r flywheel -f - >/dev/null && echo "keycloak: created client $c"
  done
  "${kc[@]}" update realms/flywheel -s loginTheme=flywheel >/dev/null 2>&1 || echo "warning: could not set the login theme" >&2
  # Group admins administers the flywheel realm in its own admin console (/auth/admin/flywheel/console/).
  "${kc[@]}" add-roles -r flywheel --gname admins --cclientid realm-management --rolename realm-admin >/dev/null 2>&1 \
    || echo "warning: could not give group admins the realm-admin role" >&2
}

# MLflow (mlflow-oidc-auth): wait until the server answers. Inside the compose network MLflow serves at the
# root; the /mlflow prefix only exists through Caddy (X-Forwarded-Prefix).
bootstrap_mlflow() {
  for _ in $(seq 1 60); do   # up to 2 min, untuned; if it runs out, run `up` again to mint the token
    "${compose[@]}" exec -T gitea curl -fs -H "Host: $SERVICE_HOST" http://mlflow:5000/health >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "warning: mlflow is not answering; access tokens can't be minted" >&2
}

# MLflow access token for API clients (MLFLOW_TRACKING_USERNAME=<name>, MLFLOW_TRACKING_PASSWORD=<token>):
# sign the user in with a Keycloak password grant, then (re)issue their "default" token, valid a year (untuned;
# `up` re-mints the jobs' token each time).
mlflow_token() {
  local name=$1 pw=$2 tok
  tok=$("${compose[@]}" exec -T gitea curl -fs -d grant_type=password -d client_id=mlflow -d "client_secret=$KC_MLFLOW_SECRET" \
        -d "username=$name" -d "password=$pw" -d scope=openid http://keycloak:8080/auth/realms/flywheel/protocol/openid-connect/token \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])') || return 1
  "${compose[@]}" exec -T gitea curl -fs -X PATCH -H "Authorization: Bearer $tok" -H "Host: $SERVICE_HOST" -H 'Content-Type: application/json' \
    -d "{\"expiration\":\"$(date -u -d '+365 days' +%Y-%m-%dT%H:%M:%SZ)\"}" http://mlflow:5000/api/2.0/mlflow/users/access-token \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])'
}

# Environments (RoboDojo eval env, policy envs) are built by the Gitea workflow setup-envs, which runs
# gitea/setup/*.sh in a job container on this node. `up` dispatches it when an env is missing
# or was built for another checkout or STATE_DIR (each installer writes both paths to <env>/.flywheel-setup);
# `ctl.sh setup` dispatches it unconditionally. Follow the run in Gitea > Actions.
setup_envs() {
  local force=${1:-} project api="http://localhost:3000/gitea/api/v1" auth="$ADMIN_USER:$ADMIN_PASSWORD" e stale=()
  project=$(cd "$here/.." && pwd)/eval/system1/RoboDojo
  for e in robodojo act dp; do
    [[ $(cat "$STATE_DIR/envs/$e/.flywheel-setup" 2>/dev/null) == "$project $STATE_DIR/envs/$e" ]] || stale+=("$e")
  done
  if [[ -z $force && ${#stale[@]} == 0 ]]; then echo "setup: environments are current for $project"; return 0; fi
  "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X POST -H 'Content-Type: application/json' -d '{"ref":"main"}' \
    "$api/repos/$ADMIN_USER/pipelines/actions/workflows/setup-envs.yml/dispatches" >/dev/null \
    && echo "setup: dispatched setup-envs (${force:+forced; }missing or stale: ${stale[*]:-none}); follow it at https://$SERVICE_HOST:$CADDY_PORT/gitea/$ADMIN_USER/pipelines/actions" \
    || echo "warning: could not dispatch setup-envs" >&2
}

# Add a person to every service under one username:
#   ctl.sh user add <name> <email> [password]
# Keycloak (SSO: Gitea, Grafana, MLflow, FiftyOne, Rerun, Loki), an MLflow
# access token for API clients (printed once) and the S3 gateway (an access key
# pair, printed once). Password defaults to a random one, printed.
# Grafana creates its own local admin (email admin@localhost). Give it the SSO admin's
# email so the Keycloak login maps onto it instead of trying to create a second "admin".
bootstrap_grafana() {
  local api="http://grafana:3000/grafana/api" auth="$ADMIN_USER:$ADMIN_PASSWORD"
  for _ in $(seq 1 60); do   # up to 2 min, untuned
    "${compose[@]}" exec -T gitea curl -fs "$api/health" >/dev/null 2>&1 && break
    sleep 2
  done
  # Once the admin has signed in through Keycloak, Grafana marks the user external and refuses edits; by then the email already matches.
  "${compose[@]}" exec -T gitea curl -fs -u "$auth" "$api/users/1" | grep -q "\"email\":\"$ADMIN_USER@flywheel.local\"" && return 0
  "${compose[@]}" exec -T gitea curl -fs -u "$auth" -X PUT -H 'Content-Type: application/json' \
    -d "{\"login\":\"$ADMIN_USER\",\"email\":\"$ADMIN_USER@flywheel.local\",\"name\":\"$ADMIN_USER\"}" "$api/users/1" >/dev/null \
    || echo "warning: could not set the Grafana admin email (SSO login as $ADMIN_USER may fail)" >&2
}

# S3 access: every gateway user (role `user`, from `user add`) reads and writes the shared buckets below, except the
# upload-only users in versitygw/upload-only.txt, who may only upload into their prefix. Everything else (mlflow,
# triton) has no policy, so only the root key (ADMIN_USER) reaches it. Rewritten on every `up`, `user add` and
# `s3-policies`.
S3_SHARED_BUCKETS="raw processed"
apply_s3_policies() {
  local users
  users=$("${compose[@]}" exec -T versitygw versitygw admin -a "$ADMIN_USER" -s "$ADMIN_PASSWORD" -er http://localhost:7070 list-users \
          | awk 'NR > 2 && $2 == "user" {print $1}' | paste -sd,)   # NR > 2: skip the table's two header lines
  "${compose[@]}" exec -T -e USERS="$users" -e BUCKETS="$S3_SHARED_BUCKETS" -e UPLOAD_ONLY="$(grep -v '^\s*#' versitygw/upload-only.txt 2>/dev/null)" \
    -e AWS_ACCESS_KEY_ID="$ADMIN_USER" -e AWS_SECRET_ACCESS_KEY="$ADMIN_PASSWORD" -e AWS_DEFAULT_REGION="${S3_REGION:-us-east-1}" mlflow python -c '
import boto3, json, os, sys
s3 = boto3.client("s3", endpoint_url="http://versitygw:7070")
upload = [l.split() for l in os.environ["UPLOAD_ONLY"].splitlines() if l.strip()]   # [user, bucket/prefix]
restricted = {u for u, _ in upload}
users = [u for u in os.environ["USERS"].split(",") if u and u not in restricted]
buckets = set(os.environ["BUCKETS"].split()) | {t.split("/", 1)[0] for _, t in upload}
for b in sorted(buckets):
    st = []
    if users and b in os.environ["BUCKETS"].split():
        st.append({"Effect": "Allow", "Principal": {"AWS": users},
                   "Action": ["s3:ListBucket", "s3:ListBucketMultipartUploads", "s3:GetObject", "s3:PutObject", "s3:DeleteObject",
                              "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
                   "Resource": [f"arn:aws:s3:::{b}", f"arn:aws:s3:::{b}/*"]})
    for u, t in upload:
        tb, prefix = (t.split("/", 1) + [""])[:2]
        if tb != b:
            continue
        prefix = prefix.strip("/")
        st.append({"Effect": "Allow", "Principal": {"AWS": [u]},
                   "Action": ["s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"],
                   "Resource": [f"arn:aws:s3:::{b}/{prefix}/*"]})
        st.append({"Effect": "Allow", "Principal": {"AWS": [u]}, "Action": ["s3:ListBucket"], "Resource": [f"arn:aws:s3:::{b}"],
                   "Condition": {"StringLike": {"s3:prefix": [prefix, f"{prefix}/*"]}}})
    if st:
        s3.put_bucket_policy(Bucket=b, Policy=json.dumps({"Version": "2012-10-17", "Statement": st}))
    else:
        s3.delete_bucket_policy(Bucket=b)
print("s3:", len(users), "users read/write", os.environ["BUCKETS"] + ";", len(upload), "upload-only", file=sys.stderr)
for b in sorted(x["Name"] for x in s3.list_buckets()["Buckets"]):   # what the gateway now holds, for the local copy
    try:
        print(b, json.dumps(json.loads(s3.get_bucket_policy(Bucket=b)["Policy"])))
    except s3.exceptions.ClientError:
        print(b, "none")' | apply_s3_policies_save
}

# Keep a copy of every bucket's policy in versitygw/policies/<bucket>.json (gitignored); none = no policy.
apply_s3_policies_save() {
  local dir=versitygw/policies b p
  mkdir -p "$dir" && rm -f "$dir"/*.json
  while read -r b p; do
    [[ $p == none ]] || python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]), indent=2))' "$p" > "$dir/$b.json"
  done
  echo "s3: policies saved in services/$dir/"
}

user_add() {
  # Random password: 18 bytes in base64 (24 chars) minus /+=, cut to 20 chars; length untuned.
  local name=${1:?name} email=${2:?email} pw=${3:-$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)}
  local kc=(docker compose --project-name flywheel exec -T keycloak /opt/keycloak/bin/kcadm.sh)
  "${kc[@]}" config credentials --server http://localhost:8080/auth --realm master --user "$ADMIN_USER" --password "$ADMIN_PASSWORD" >/dev/null
  if "${kc[@]}" get users -r flywheel -q "username=$name" --fields username 2>/dev/null | grep -q "\"$name\""; then
    echo "keycloak: $name exists"
  else
    "${kc[@]}" create users -r flywheel -s "username=$name" -s "email=$email" -s enabled=true -s emailVerified=true >/dev/null
    "${kc[@]}" set-password -r flywheel --username "$name" --new-password "$pw"
    echo "keycloak: created $name"
  fi
  # Every app checks group membership (users or admins); make sure of it even for an account created elsewhere.
  local gid uid
  gid=$("${kc[@]}" get groups -r flywheel -q search=users --fields id,name | python3 -c 'import sys,json; print([g["id"] for g in json.load(sys.stdin) if g["name"]=="users"][0])')
  uid=$("${kc[@]}" get users -r flywheel -q "username=$name" --fields id,username | python3 -c 'import sys,json; print([u["id"] for u in json.load(sys.stdin) if u["username"]==sys.argv[1]][0])' "$name")
  if "${kc[@]}" get "users/$uid/groups" -r flywheel --fields name | grep -q '"\(users\|admins\)"'; then
    echo "keycloak: $name is in group users or admins"
  else
    "${kc[@]}" update "users/$uid/groups/$gid" -r flywheel -n
    echo "keycloak: added $name to group users"
  fi

  local tok
  if tok=$(mlflow_token "$name" "$pw" 2>/dev/null) && [[ -n $tok ]]; then
    echo "mlflow: access token $tok   (MLFLOW_TRACKING_USERNAME=$name MLFLOW_TRACKING_PASSWORD=<token>; shown once)"
  else
    echo "mlflow: no token minted (sign in at /mlflow with Keycloak; Profile > Tokens issues one)"
  fi

  local secret; secret=$(openssl rand -hex 20)   # 40 chars, the length of an AWS secret key
  if docker compose --project-name flywheel exec -T -e ROOT_ACCESS_KEY_ID="$ADMIN_USER" -e ROOT_SECRET_ACCESS_KEY="$ADMIN_PASSWORD" versitygw \
       versitygw admin -a "$ADMIN_USER" -s "$ADMIN_PASSWORD" -er http://localhost:7070 create-user -a "$name" -s "$secret" -r user >/dev/null 2>&1; then
    echo "s3: access key $name, secret $secret   (endpoint https://s3.$SERVICE_HOST:$CADDY_PORT; shown once)"
    apply_s3_policies
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
  for d in caddy/data caddy/config versitygw/buckets versitygw/buckets/raw versitygw/buckets/processed versitygw/buckets/processed/xpolicylab versitygw/buckets/mlflow versitygw/meta versitygw/iam mlflow loki grafana gitea/data gitea/config act_runner mongo fiftyone prometheus prometheus-sd slurm-exporter keycloak/db keycloak/import keycloak/login-background versitygw/buckets/triton/models versitygw/buckets/logging/chunks; do
    mkdir -p "$STATE_DIR/$d"
  done
  # The Slurm job pusher, where an account that sees every job can read it (it may not be able to read this checkout).
  cp -f "$here/slurm-exporter/exporter.py" "$STATE_DIR/slurm-exporter/exporter.py"
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
    for _ in $(seq 1 60); do [[ -f $ca ]] && break; sleep 1; done   # up to 1 min, untuned
    [[ -f $ca ]] || { echo "caddy did not create its CA at $ca" >&2; exit 1; }
  fi
  "${compose[@]}" up -d "${@:2}" $("${compose[@]}" config --services | grep -vx act_runner)
  bootstrap_gitea
  bootstrap_grafana
  bootstrap_keycloak
  bootstrap_mlflow
  bootstrap_jobs
  apply_s3_policies
  "${compose[@]}" up -d "${@:2}"
  keep_off_gpu_memory
  # Triton on its node, and where Prometheus finds its metrics.
  mkdir -p "$STATE_DIR/prometheus-sd"
  printf '[{"targets": ["%s:8102"], "labels": {"node": "%s"}}]\n' "$TRITON_ADDR" "$TRITON_NODE" > "$STATE_DIR/prometheus-sd/triton.json"
  printf '[{"targets": ["%s:8199"], "labels": {"node": "%s"}}]\n' "$LLM_GATEWAY_ADDR" "${DYNAMO_HEAD:-}" > "$STATE_DIR/prometheus-sd/llm-gateway.json"
  triton_compose up -d
  setup_envs
  exit 0
fi

if [[ $1 == dynamo ]]; then
  dynamo "${@:2}"
  exit 0
fi

if [[ $1 == triton ]]; then
  triton_compose "${@:2}"
  exit 0
fi

if [[ $1 == s3-policies ]]; then
  apply_s3_policies
  exit 0
fi

if [[ $1 == setup ]]; then
  setup_envs force
  exit 0
fi

exec "${compose[@]}" "$@"
