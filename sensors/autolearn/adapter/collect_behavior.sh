#!/usr/bin/env bash
# Collect BEHAVIOR-1K demonstrations: run one ASPIRE policy on a range of task instances, one Isaac Sim process at a
# time, until a number of successful episodes exist. Every run keeps ASPIRE's full artifacts (states and actions per
# tick, three videos); behavior_to_lerobot.py turns the successful ones into a dataset. Success is outcome.json's
# task_success (the pickup witness) or, where a task's witness writes only protocol_success (the door tasks), that.
#
#   adapter/collect_behavior.sh <policy.py> <first seed> <last seed> [--target N] [--repeats R] [--suite turning_on_radio] [--out DIR] [--grace TICKS]
#
# Seeds are task-instance ids (0-299 exist); a campaign's evaluation seeds (1-25) are left alone by choosing 36 up.
# Each seed is run up to R times (default 1); the loop stops at N successes (default 100). Outcomes append to
# <out>/collect.jsonl, so an interrupted collection resumes by rerunning with the same arguments.
set -euo pipefail
A="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
policy="$(realpath "$1")"; first=$2; last=$3; shift 3
target=100; repeats=1; suite=turning_on_radio; out=""; grace=300
while [[ $# -gt 0 ]]; do
  case $1 in
    --target) target=$2; shift 2 ;;
    --repeats) repeats=$2; shift 2 ;;
    --suite) suite=$2; shift 2 ;;
    --out) out=$2; shift 2 ;;
    --grace) grace=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
out="$(realpath -m "${out:-$A/aspire/outputs/collect/behavior/$suite/$(basename "${policy%.py}")}")"   # absolute: the runs happen from aspire/
mkdir -p "$out"
log="$out/collect.jsonl"
cd "$A/aspire"
export OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=${OMNIGIBSON_GPU_ID:-0} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$A/behavior-data}" OMNIGIBSON_APPDATA_PATH="${OMNIGIBSON_APPDATA_PATH:-$A/aspire/.behavior-appdata}"
harness="${BEHAVIOR_HARNESS:-$A/cap-harness-behavior.sh}"   # gb300/collect_behavior.sbatch runs run_behavior.py unsandboxed
ulimit -c 0
# ticks the episode runs on after the witness first succeeds, so a demonstration includes the lift (run_behavior.py);
# 300 (10 s) covers the policy's 3 cm lift steps to 15 cm, and the program usually ends before it
export CAP_HARNESS_SUCCESS_GRACE_TICKS=$grace

successes() { [[ -f $log ]] && grep -c '"task_success": true' "$log" || true; }   # grep -c prints 0 itself on no match
done_runs() { [[ -f $log ]] && grep -c "\"seed\": $1, \"repeat\": $2," "$log" || true; }

for seed in $(seq "$first" "$last"); do
  for rep in $(seq 1 "$repeats"); do
    [[ $(successes) -ge $target ]] && { echo "$(successes) successes in $log"; exit 0; }
    [[ $(done_runs "$seed" "$rep") -gt 0 ]] && continue
    # python processes only: a shell whose command line merely mentions these names must not count. Not under Slurm,
    # where each job has a GPU of its own and other jobs' runs share the node
    if [[ -z ${SLURM_JOB_ID:-} ]] && pgrep -f 'python[^ ]* [^ ]*(run_behavior\.py|geniesim_benchmark/app/app\.py)' > /dev/null; then
      echo "another Isaac Sim process is running; stopping" >&2; exit 1
    fi
    root="$out/seed_$(printf '%04d' "$seed")/rep_$rep"
    mkdir -p "$root"
    "$harness" run --benchmark behavior --suite "$suite" --task-id 0 --seed "$seed" --program "$policy" \
      --output-root "$root" --flat-run-dir --max-steps 6000 --camera-width 512 --camera-height 512 > "$root/run.log" 2>&1 || true
    outcome="$(find "$root" -name outcome.json | head -1)"
    if [[ -z $outcome ]]; then
      echo "{\"seed\": $seed, \"repeat\": $rep, \"run\": null, \"task_success\": null, \"note\": \"no outcome (launch failure)\"}" >> "$log"
      continue
    fi
    python3 - "$seed" "$rep" "$outcome" "$log" <<'PY'
import json, sys, os
seed, rep, path, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
o = json.load(open(path))
rec = {"seed": seed, "repeat": rep, "run": os.path.dirname(path), "task_success": bool(o["task_success"] if o.get("task_success") is not None else o.get("protocol_success")),
       "steps": o["steps_executed"], "termination": o["termination_reason"], "wall_s": round(o["timing"]["wall_s"])}
open(log, "a").write(json.dumps(rec) + "\n")
print(rec, flush=True)
PY
  done
done
echo "$(successes) successes in $log (seeds $first-$last exhausted)"
