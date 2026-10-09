#!/usr/bin/env bash
# Run IsaacLab-Arena agentic environment generation through a local OpenAI-compatible proxy (CLIProxyAPI).
#   bash sim/isaaclab_arena/envgen.sh cli --mode resolve --prompt "..."   # CLI runner (writes YAML)
#   bash sim/isaaclab_arena/envgen.sh cli --mode build --env_spec eval/system2/environments/<env>.yaml
#   bash sim/isaaclab_arena/envgen.sh gui                                  # live editor on http://localhost:8501
# Generated specs go to eval/system2/environments/ unless --out_dir is given.
# Env: OPENAI_API_KEY (proxy client key), ARENA_PROXY_BASE_URL (default http://127.0.0.1:8317/v1),
#      ARENA_PROXY_MODEL (default claude-sonnet-5-5; CLI also accepts --model),
#      ARENA_SPLAT_SCENE (a sim/splat scene.usda, registered as the `splat_scene` background).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARENA="${ROOT}/eval/system2/IsaacLab-Arena"
OUT_DIR="${ROOT}/eval/system2/environments"

mode="${1:?usage: envgen.sh cli|gui [args...]}"; shift
case "${mode}" in
  cli) runner=cli_runner.py ;;
  gui) runner=gui_runner.py ;;
  *) echo "usage: envgen.sh cli|gui [args...]" >&2; exit 2 ;;
esac

# Resolve user paths before changing into the Arena checkout.
args=()
for a in "$@"; do
  if [[ "${a}" != -* && -e "${a}" ]]; then args+=("$(realpath "${a}")"); else args+=("${a}"); fi
done
if [[ " ${args[*]} " != *" --out_dir "* && ( "${mode}" == gui || " ${args[*]} " == *" --mode resolve "* || " ${args[*]} " == *" --mode full "* ) ]]; then
  args+=(--out_dir "${OUT_DIR}")
fi
if [[ "${mode}" == cli && " ${args[*]} " != *" --inference_endpoint "* ]]; then
  args+=(--inference_endpoint cliproxy)
fi

cd "${ARENA}"
export PYTHONPATH="${ROOT}/sim/isaaclab_arena/hooks${PYTHONPATH:+:${PYTHONPATH}}"
export ARENA_INFERENCE_ENDPOINT=cliproxy OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export STREAMLIT_SERVER_HEADLESS=true STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-localhost}"   # no email prompt; local-only unless overridden
[[ $(uname -m) == aarch64 ]] && export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/lib/aarch64-linux-gnu/libgomp.so.1"   # Isaac Sim on ARM refuses to start without it
exec .venv/bin/python "isaaclab_arena_examples/agentic_environment_generation/${runner}" "${args[@]}"
