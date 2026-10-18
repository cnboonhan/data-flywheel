#!/usr/bin/env bash
# Install the tasks under this directory into the Genie Sim benchmark without editing upstream code:
# copy each task's config yaml, eval_tasks json and instance folders to where geniesim_benchmark reads
# them, and list those paths in the submodule's git exclude file so `git status` stays clean.
#
# Copies, not symlinks: adapter/ is not mounted in the geniesim3 container (only genie_sim/ is, as
# /workspace), so a link out of the submodule would dangle there. Re-run after regenerating a task.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
G="$(cd "$HERE/../.." && pwd)/genie_sim"
PKG=source/geniesim_benchmark/src/geniesim_benchmark
B="$G/$PKG"
EXCLUDE="$(git -C "$G" rev-parse --path-format=absolute --git-path info/exclude)"  # the superproject's .git/modules/... copy

exclude() {  # append a path (relative to the submodule root) once
  grep -qxF "$1" "$EXCLUDE" 2>/dev/null || echo "$1" >> "$EXCLUDE"
}

for task_dir in "$HERE"/*/; do
  task="$(basename "$task_dir")"
  [[ -d "$task_dir/instances" ]] || continue
  yaml="$(ls "$task_dir"/*.yaml)"
  json="$(ls "$task_dir"/*.json)"
  cp "$yaml" "$B/config/"
  cp "$json" "$B/benchmark/config/eval_tasks/"
  rm -rf "$B/benchmark/config/llm_task/$task"
  cp -r "$task_dir/instances" "$B/benchmark/config/llm_task/$task"
  exclude "$PKG/config/$(basename "$yaml")"
  exclude "$PKG/benchmark/config/eval_tasks/$(basename "$json")"
  exclude "$PKG/benchmark/config/llm_task/$task/"
  echo "installed $task: $(basename "$yaml"), $(basename "$json"), $(ls "$task_dir/instances" | wc -l) instances"
done

status="$(git -C "$G" status --short)"
if [[ -n "$status" ]]; then
  echo "genie_sim is not clean after install:" >&2
  echo "$status" >&2
  exit 1
fi
echo "genie_sim clean (excludes in $EXCLUDE)"
