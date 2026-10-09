#!/usr/bin/env bash
# Build IsaacLab-Arena's uv .venv (Isaac Sim 6.1) on x86_64 or aarch64, without committing anything in the submodule.
#   bash eval/system2/setup-arena.sh
# Upstream locks for Linux x86_64 only, and its lockfile is stale. Isaac Sim 6.1 also ships aarch64 wheels, so this
# adds the current machine's platform to [tool.uv] environments, regenerates uv.lock, and hides both local edits
# with skip-worktree (undo with --no-skip-worktree before pulling Arena). It also overrides NVRTC to >= 12.9 (see below). CMAKE_POLICY_VERSION_MINIMUM lets CMake 4
# build egl-probe (robomimic's, sdist only, with an old cmake_minimum_required).
set -euo pipefail
ARENA="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/IsaacLab-Arena"
git -C "${ARENA}/../../.." submodule update --init eval/system2/IsaacLab-Arena
git -C "${ARENA}" submodule update --init --recursive
cd "${ARENA}"
git update-index --no-skip-worktree pyproject.toml uv.lock

python3 - "$(uname -m)" <<'PY'
import re, sys
arch, path = sys.argv[1], "pyproject.toml"
s = open(path).read()
m = re.search(r"^environments = \[(.*)\]$", s, re.M)
marker = f"sys_platform == 'linux' and platform_machine == '{arch}'"
if m and marker not in m.group(1):
    s = s[:m.start(1)] + m.group(1) + f', "{marker}"' + s[m.end(1):]
    open(path, "w").write(s)
    print(f"pyproject.toml: added {arch} to [tool.uv] environments")
# torch's cu128 wheels pin NVRTC 12.8, which can't compile for sm_103 (B300/GB300): torch's runtime-compiled kernels
# then fail with "invalid value for --gpu-architecture". NVRTC 12.9 (same soname) knows it and works everywhere else.
nvrtc = '"nvidia-cuda-nvrtc-cu12>=12.9",'
if nvrtc not in s:
    s = s.replace("override-dependencies = [\n", "override-dependencies = [\n    " + nvrtc + "\n", 1)
    open(path, "w").write(s)
    print("pyproject.toml: NVRTC >= 12.9 override")
PY
uv lock
git update-index --skip-worktree pyproject.toml uv.lock
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --frozen --extra dev
[[ $(uname -m) == aarch64 ]] && export LD_PRELOAD="${LD_PRELOAD:+${LD_PRELOAD}:}/lib/aarch64-linux-gnu/libgomp.so.1"   # Isaac Sim on ARM refuses to start without it
OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y .venv/bin/python -c "import isaacsim, isaaclab_arena; print('arena venv ready')" </dev/null
