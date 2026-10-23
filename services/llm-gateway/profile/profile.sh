#!/usr/bin/env bash
# How much Caddy and the LLM gateways add to one model's latency: services/llm-gateway/profile/profile.sh <model>
# Runs bench.py on the node that serves the model, once per path, from the outside in: the public URL (Caddy, the
# head's gateway, the model node's gateway when it is another node), each gateway directly, then the model's Dynamo
# frontend. Only the hops differ between rows, as every request starts on the same node. Prints a Markdown table.
set -euo pipefail
model=${1:?usage: profile.sh <model>}
here=$(cd "$(dirname "$0")" && pwd)
services=$(cd "$here/../.." && pwd)
eval "$(grep -E '^(STATE_DIR|DYNAMO_HEAD)=' "$services/.env" | sed 's/^/export /')"
set -a; . "$STATE_DIR/flywheel.env"; set +a   # LLM_URL, and the stack's CA for it
ca=$MLFLOW_TRACKING_SERVER_CERT_PATH
cfg="$STATE_DIR/dynamo"

# Where the model is served: the node whose gateway config has its frontend (render.py writes one per node).
read -r node frontend < <(python3 - "$cfg" "$model" <<'EOF'
import glob, json, os, sys
for f in sorted(glob.glob(f"{sys.argv[1]}/*.gateway.json")):
    m = json.load(open(f))["models"].get(sys.argv[2], {})
    if "frontend" in m:
        print(os.path.basename(f).removesuffix(".gateway.json"), m["frontend"]); break
EOF
) || true   # nothing found: read fails, and the check below says why
[[ -n ${node:-} ]] || { echo "$model is not served now (no gateway config has its frontend)" >&2; exit 1; }
addr() { getent hosts "$1" | awk '{print $1; exit}'; }

paths=("full path (Caddy, gateways)|${LLM_URL%/v1}|$ca" "head gateway ($DYNAMO_HEAD)|http://$(addr "$DYNAMO_HEAD"):8199|")
[[ $node != "$DYNAMO_HEAD" ]] && paths+=("node gateway ($node)|http://$(addr "$node"):8199|")
paths+=("Dynamo frontend|$frontend|")

echo "$model, served on $node; every request from $node"
echo
echo "| Path | First token | Total | Decode |"
echo "|---|---|---|---|"
for p in "${paths[@]}"; do
  IFS='|' read -r label url cafile <<<"$p"
  ssh -o BatchMode=yes "$node" "python3 -I $(printf '%q ' "$here/bench.py" "$label" "$url" "$model" $cafile)"
done
