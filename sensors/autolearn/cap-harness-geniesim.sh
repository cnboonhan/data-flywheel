#!/usr/bin/env bash
# The Genie Sim runner (adapter/run_geniesim.py, ASPIRE's harness on the benchmark's policy socket) in the same
# sandbox as cap-harness-sandboxed.sh, so agent-written programs can run on Genie Sim. Same arguments as the runner.
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAP_HARNESS_BIN="$here/aspire/.venv-pyroki/bin/python" exec "$here/cap-harness-sandboxed.sh" "$here/adapter/run_geniesim.py" "$@"
