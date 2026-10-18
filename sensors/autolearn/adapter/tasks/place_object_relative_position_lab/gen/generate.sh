#!/usr/bin/env bash
# Regenerate the four instances of place_object_relative_position_lab with geniesim_generator (CPU only,
# in the geniesim3 container). Overwrites ../instances/. Object ids are re-drawn on every run (the DSL
# stamps each call with uuid4), so checks are rewritten too.
#
# What the container lacks and how it is bridged (nothing is downloaded):
#   - no `python`: the container has `python3` (with pxr) and `omni_python` (without the generator's deps)
#   - no mitsuba / pygraphviz / jaxtyping / transforms3d: stand-ins in shims/, first on PYTHONPATH
#   - app.py imports LLM_RESULT.py from its own directory, so it is run through runpy from a directory
#     holding ours; the package's template is never touched
#   - adapter/ is not mounted in the container: gen/ is staged under genie_sim/.lab-gen (and removed)
# Output is read back from <generator>/src/benchmark/config/llm_task/<scene_id>/<n>/ (removed after).
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
A="$(cd "$HERE/../../../.." && pwd)"              # sensors/autolearn
G="$A/genie_sim"
SCENE_ID=place_object_relative_position_lab
STAGE="$G/.lab-gen"                              # /workspace/.lab-gen in the container
GEN_PKG=/workspace/source/geniesim_generator/src/geniesim_generator
OUT_HOST="$G/source/geniesim_generator/src/benchmark/config/llm_task/$SCENE_ID"
OUT_CONT="/workspace/source/geniesim_generator/src/benchmark/config/llm_task/$SCENE_ID"
NUM_INSTANCES=4

run() {  # run a python3 command in the container as the invoking user, GPU hidden (another Isaac process owns it)
  docker exec -u "$(id -u):$(id -g)" -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH="/workspace/.lab-gen/shims:$GEN_PKG:/workspace/source/geniesim_generator/src:/" "$@"
}
cleanup() { rm -rf "$STAGE" "$G/source/geniesim_generator/src/benchmark"; }
trap cleanup EXIT

cleanup
cp -r "$HERE" "$STAGE"
for n in $(seq 0 $((NUM_INSTANCES - 1))); do
  echo "== instance $n"
  run -e LAB_INSTANCE="$n" geniesim3 bash -lc "cd /workspace/.lab-gen && python3 -c '
import runpy, sys
sys.argv = [\"app.py\", \"--scene_id\", \"$SCENE_ID\"]
runpy.run_path(\"$GEN_PKG/app.py\", run_name=\"__main__\")'" 2>&1 | grep -E '^(step3|Random seed|Added payload)' || true
  [[ -f "$OUT_HOST/$n/scene.usda" ]] || { echo "instance $n was not written"; exit 1; }
  cp "$HERE/LLM_RESULT.py" "$OUT_HOST/$n/LLM_RESULT.py"   # app.py snapshots the package's template, not ours
done
echo "== payload paths and kinematic table"
run geniesim3 python3 /workspace/.lab-gen/postprocess.py "$OUT_CONT"
echo "== instructions and checks"
python3 "$HERE/checks.py" "$OUT_HOST"
echo "== verify"
run geniesim3 python3 /workspace/.lab-gen/verify.py "$OUT_CONT"
rm -rf "$HERE/../instances"
cp -r "$OUT_HOST" "$HERE/../instances"
echo "wrote $(cd "$HERE/.." && pwd)/instances"
