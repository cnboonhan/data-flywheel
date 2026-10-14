#!/usr/bin/env bash
# Write nodes.json, the GPU nodes Prometheus scrapes: every node of every Slurm partition, by IP (containers on the
# service node can't resolve the cluster's host names), labelled with its name and partition. Run it on a login node
# after nodes change; Prometheus rereads the file by itself.
set -euo pipefail
out="$(dirname "$0")/nodes.json"
sinfo -a -h -o "%P %N" | while read -r part list; do
  for n in $(scontrol show hostnames "$list"); do
    ip=$(getent hosts "$n" | awk '{print $1; exit}')
    [[ -n $ip ]] && printf '%s %s %s\n' "${part%\*}" "$n" "$ip"
  done
done | sort -k2 -u | python3 -c '
import json, sys
rows = [l.split() for l in sys.stdin]
print(json.dumps([{"targets": [ip], "labels": {"node": n, "partition": p}} for p, n, ip in rows], indent=1))
' > "$out"
echo "$(grep -c '"node"' "$out") nodes -> $out"
