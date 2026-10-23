#!/usr/bin/env bash
# Collect BEHAVIOR-1K demonstrations over many GB300 GPUs: K collect_behavior.sbatch jobs (SAM3 + simulator each) over
# disjoint seed ranges, then aggregate_behavior.sbatch once they all end (converts the successes, uploads to raw).
#
#   gb300/collect_parallel.sh <policy.py> <first seed> <last seed> <jobs> <target> [collect_behavior.sh options]
#   e.g. gb300/collect_parallel.sh adapter/programs/pick_cup_r1pro.py 46 299 16 500 --suite picking_up_cup
#
# Each job stops at ceil(target / ranges) successes (ranges: the seed ranges submitted, up to <jobs>) or the end of
# its range; a run is ~3-8 min, so size the range to the policy's success rate (a seed is tried once per --repeats).
set -euo pipefail
A="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
policy=$(realpath "$1"); first=$2; last=$3; jobs=$4; target=$5; shift 5
suite=picking_up_cup; args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do [[ ${args[i]} == --suite ]] && suite=${args[i+1]}; done
n=$((last - first + 1)); per=$(( (n + jobs - 1) / jobs ))
ranges=$(( (n + per - 1) / per )); each=$(( (target + ranges - 1) / ranges ))   # the ranges, not the jobs asked for, share the target
ids=()
for ((s = first; s <= last; s += per)); do
  e=$((s + per - 1 > last ? last : s + per - 1))
  ids+=("$(sbatch --parsable "$A/gb300/collect_behavior.sbatch" "$policy" "$s" "$e" --target "$each" "${args[@]}")")
  echo "collect ${ids[-1]}: seeds $s-$e, up to $each successes"
done
agg=$(sbatch --parsable --dependency="afterany:$(IFS=:; echo "${ids[*]}")" "$A/gb300/aggregate_behavior.sbatch" \
  "$suite" "$(basename "${policy%.py}")")
echo "aggregate $agg: after ${ids[*]}"
