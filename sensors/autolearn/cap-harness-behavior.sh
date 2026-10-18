#!/usr/bin/env bash
# cap-harness for BEHAVIOR-1K: adapter/run_behavior.py (PyRoki in place of cuRobo) in the same sandbox as
# cap-harness-sandboxed.sh. Set CAP_HARNESS to this script for the BEHAVIOR agent loop.
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CAP_HARNESS_BIN="$here/aspire/.venv-behavior/bin/python" exec "$here/cap-harness-sandboxed.sh" "$here/adapter/run_behavior.py" "$@"
