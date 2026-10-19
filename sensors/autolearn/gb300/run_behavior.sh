#!/usr/bin/env bash
# adapter/run_behavior.py in the GB300 BEHAVIOR environment, without cap-harness-behavior.sh's bubblewrap sandbox (the
# nodes have no bwrap): only for reviewed programs. collect_behavior.sbatch passes it to the collector.
# Each run gets its own IPC namespace and /dev/shm (an unprivileged user namespace): Isaac processes of one user on one
# node otherwise share named IPC, and concurrent runs hang after "app ready" or fail to load materials.
# BEHAVIOR_RUN_TIMEOUT (s, default 1800: a pickup run is 2-5 min plus up to ~5 min of cold scene load) ends a run
# that hangs, so it costs one seed, not the job.
A="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
py="${STATE:-/tier1/mistral/flywheel/autolearn}/venv-behavior/bin/python"
exec timeout -k 30 "${BEHAVIOR_RUN_TIMEOUT:-1800}" unshare --user --map-root-user --ipc --mount -- \
  bash -c 'mount -t tmpfs -o size=32g tmpfs /dev/shm && exec "$0" "$@"' "$py" "$A/adapter/run_behavior.py" "$@"
