#!/bin/bash
# Split an evaluation's instances over several one-GPU jobs, as collect_parallel.sh does for collection.
#   [EVAL_SBATCH_OPTS=--dependency=afterok:<training job>] \
#     gb300/eval_parallel.sh <jobs> <eval sbatch> <its arguments before the seeds...> -- <seed> [<seed> ...]
#   e.g. gb300/eval_parallel.sh 6 gb300/eval_pi05.sbatch after_v1 $STATE/pi05/.../19999 $STATE/collect/.../pick_cup_heldout \
#          -- $(cat $STATE/eval/galaxea/heldout_seeds.txt)
# Every job gets the same label, so all of them append to one results.jsonl; each skips the instances already in it,
# so rerunning the same command resumes what is missing. Seeds are dealt round-robin, so each job gets a spread of
# them rather than one block.
set -euo pipefail
jobs=$1 sbatch_file=$2; shift 2
fixed=()
while [[ $# -gt 0 && $1 != -- ]]; do fixed+=("$1"); shift; done
[[ ${1:-} == -- ]] || { echo "usage: eval_parallel.sh <jobs> <eval sbatch> <args...> -- <seeds...>" >&2; exit 2; }
shift
seeds=("$@")
read -ra opts <<< "${EVAL_SBATCH_OPTS:-}"   # extra sbatch options for every job, e.g. wait for the training job
(( jobs > ${#seeds[@]} )) && jobs=${#seeds[@]}
for ((j = 0; j < jobs; j++)); do
  mine=()
  for ((i = j; i < ${#seeds[@]}; i += jobs)); do mine+=("${seeds[i]}"); done
  echo "job $((j + 1))/$jobs: seeds ${mine[*]} -> $(sbatch --parsable "${opts[@]}" "$sbatch_file" "${fixed[@]}" "${mine[@]}")"
done
