#!/usr/bin/env bash
# cap-harness with the user's credentials hidden (empty tmpfs over each directory, /dev/null over each file):
# ASPIRE runs model-generated programs inside it. Set CAP_HARNESS to this script for the agent loops.
# CAP_HARNESS_BIN picks the binary (default: the LIBERO venv's cap-harness; cap-harness-behavior.sh sets its own).
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bin="${CAP_HARNESS_BIN:-$here/aspire/.venv-libero/bin/cap-harness}"
hide=()
for d in ~/.aws ~/.ssh ~/.claude ~/.config ~/.docker ~/.gnupg ~/.kube ~/.cache/huggingface ~/.local/share/keyrings; do
  [[ -d $d ]] && hide+=(--tmpfs "$d")
done
for f in ~/.netrc ~/.git-credentials ~/.claude.json ~/.pypirc; do
  [[ -f $f ]] && hide+=(--ro-bind /dev/null "$f")
done
exec bwrap --dev-bind / / "${hide[@]}" -- "$bin" "$@"
