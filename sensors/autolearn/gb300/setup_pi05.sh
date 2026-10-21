#!/usr/bin/env bash
# π0.5 (XPolicyLab's policy/Pi_05: Physical Intelligence's openpi, JAX) on the GB300 (arm64), under $STATE, idempotent.
# openpi's own `uv sync` would build its venv inside the submodule; instead its lock is exported (with the lerobot group)
# and installed into $STATE/venv-pi05, then openpi and openpi-client themselves, not editable, so nothing is written
# into the submodule. Also downloads the π0.5 base checkpoint (gs://openpi-assets, Apache-2.0 code, weights under the
# Gemma Terms of Use) through openpi's own downloader into $STATE/models/openpi.
#   sensors/autolearn/gb300/setup_pi05.sh
set -euo pipefail
A="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATE=${STATE:-/tier1/mistral/flywheel/autolearn}
O=$A/../../eval/system1/RoboDojo/XPolicyLab/policy/Pi_05/openpi
V=$STATE/venv-pi05
PIP=(uv pip install --python "$V/bin/python")
if [[ ! -f $V/.pi05-ok ]]; then
  uv venv --allow-existing --python 3.11 "$V"
  (cd "$O" && uv export --frozen --no-dev --group lerobot --no-emit-project --no-emit-workspace --no-hashes -o "$STATE/pi05/requirements.txt")
  # the lock is a complete, pinned set (with openpi's overrides applied): install it as is, no re-resolution
  # minus torchcodec: no Linux arm64 wheels; lerobot decodes video with PyAV (bundled FFmpeg) instead
  grep -v -E '^torchcodec==' "$STATE/pi05/requirements.txt" > "$STATE/pi05/requirements-arm64.txt"
  "${PIP[@]}" --no-deps -r "$STATE/pi05/requirements-arm64.txt" --index-strategy unsafe-best-match
  "${PIP[@]}" --no-deps "$O/packages/openpi-client" "$O"
  "${PIP[@]}" pytest    # imported by openpi.models_pytorch, a dev dependency the --no-dev export leaves out
  rm -rf "$O/build" "$O/packages/openpi-client/build" "$O"/src/*.egg-info "$O"/packages/openpi-client/src/*.egg-info
  "$V/bin/python" -c 'import openpi.models.model, jax, lerobot, torch; print("openpi ok, jax", jax.__version__, "lerobot", lerobot.__version__, "torch", torch.__version__)' \
    && touch "$V/.pi05-ok"
fi
# the π0.5 base checkpoint
export OPENPI_DATA_HOME=$STATE/models/openpi
"$V/bin/python" -c 'from openpi.shared import download; print(download.maybe_download("gs://openpi-assets/checkpoints/pi05_base/params"))'
