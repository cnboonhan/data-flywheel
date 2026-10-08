# Sourced by scripts that expect conda; see ../../bin/conda.
conda() {
  case "${1:-}" in
    activate)
      local env=${2:?env}
      [[ $env == */* ]] || env="${ENVS_DIR:-/tier1/htx_boonhan/services/envs}/$env"
      [[ -f $env/bin/activate ]] || { echo "conda shim: no venv at $env" >&2; return 1; }
      source "$env/bin/activate" ;;
    deactivate)
      if type deactivate >/dev/null 2>&1; then deactivate; fi ;;
    info)
      [[ ${2:-} == --base ]] && echo "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" ;;
    *) echo "conda shim: unsupported: $*" >&2; return 2 ;;
  esac
}
