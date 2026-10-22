#!/bin/bash
# The successful runs of a BEHAVIOR-1K collection (every collect_behavior.sbatch job's seeds_*/ dir) as one LeRobot
# v2.1 dataset, uploaded to the raw bucket: s3://raw/internal_datasets/autolearn-behavior/<suite>/<policy>/.
#   sbatch sensors/autolearn/gb300/aggregate_behavior.sbatch <suite> <policy name>   (collect_parallel.sh submits it)
# The dataset is built in $STATE/datasets/behavior_<suite>_<policy> first; a rerun rebuilds it and syncs again.
#SBATCH -p raus -N1 -n1 -c16 --mem=64G -t 04:00:00 -J behavior-aggregate
#SBATCH -o /tier1/mistral/flywheel/autolearn/logs/aggregate-%j.log
set -euo pipefail
A=/home/htx_boonhan/workspaces/data-flywheel/sensors/autolearn   # sbatch runs a copy of this file
STATE=/tier1/mistral/flywheel/autolearn
suite=$1; policy=$2
src=$STATE/collect/behavior/$suite/$policy
out=$STATE/datasets/behavior_${suite}_$policy
dst=s3://raw/internal_datasets/autolearn-behavior/$suite/$policy
cat "$src"/seeds_*/collect.jsonl > "$src/collect.jsonl"   # one log over every job, for the converter and the README
echo "$(grep -c '"task_success": true' "$src/collect.jsonl") successes of $(wc -l < "$src/collect.jsonl") runs in $src"
rm -rf "$out"
"$STATE/venv-lerobot/bin/python" "$A/adapter/behavior_to_lerobot.py" "$src" --out "$out" --repo-id "autolearn-behavior/$suite" --visuals third_person
cp "$src/collect.jsonl" "$out/meta/collect.jsonl"   # which instance each episode came from, and the failures
set -a; . /tier1/mistral/flywheel/flywheel.env; set +a
AWS_ENDPOINT_URL=$S3_ENDPOINT_URL uvx --from awscli aws s3 sync --only-show-errors --delete "$out" "$dst"
echo "uploaded $out to $dst"
