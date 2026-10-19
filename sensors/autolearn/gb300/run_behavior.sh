#!/usr/bin/env bash
# adapter/run_behavior.py in the GB300 BEHAVIOR environment, without cap-harness-behavior.sh's bubblewrap sandbox (the
# nodes have no bwrap): only for reviewed programs. collect_behavior.sbatch passes it to the collector.
A="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${STATE:-/tier1/mistral/flywheel/autolearn}/venv-behavior/bin/python" "$A/adapter/run_behavior.py" "$@"
