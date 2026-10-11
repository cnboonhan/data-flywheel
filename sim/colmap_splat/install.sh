#!/usr/bin/env bash
# Install 3DGRUT (sim/colmap_splat/3dgrut, untouched upstream) into its uv .venv, on x86_64 or aarch64.
#   bash sim/colmap_splat/install.sh                 # CUDA version picked from the GPU (or CUDA_VERSION=12.8|13)
#
# Wraps upstream's scripts/create_venv_cuda.sh + install_env_uv.sh and fills two platform gaps:
#  - CUDA toolkit: upstream downloads an x86_64 runfile into .venv/cuda-<ver>/. Where no runfile exists for the
#    machine (aarch64), the same directory is assembled from NVIDIA's per-component redist tarballs, and upstream
#    then finds it. The host needs no CUDA toolkit either way.
#  - CUDA version: 12.8 by default; 13.0 for GPUs CUDA 12.8 can't compile for (sm_103, e.g. B300/GB300).
#  - USD: 3DGRUT's export needs pxr, but PyPI's usd-core has no aarch64 wheel; then OpenUSD (core + Python only)
#    is built into .venv/opt/usd (~10 min with many cores).
# Needs uv and a GCC upstream accepts; ~20 GB. Run where the target GPU is visible (or set CUDA_VERSION).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRUT="${HERE}/3dgrut"
ARCH="$(uname -m)"
USD_TAG="${USD_TAG:-v26.08}"   # usd-core 26.8, the version PyPI ships for x86_64
JOBS="${JOBS:-$(nproc)}"

if [[ -z "${CUDA_VERSION:-}" ]]; then
  CUDA_VERSION=12.8
  caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  if grep -qE '^(10\.[1-9]|11\.)' <<<"${caps}"; then CUDA_VERSION=13; fi
  echo "CUDA ${CUDA_VERSION} (GPU compute capability: ${caps:-not visible here})"
fi
export CUDA_VERSION

git -C "${HERE}/../.." submodule update --init sim/colmap_splat/3dgrut
cd "${GRUT}"
unset DISPLAY   # the CUDA runfile is a makeself archive that wants an xterm when it sees a display but no TTY
# CUDA_FULL_VERSION for this CUDA_VERSION, as upstream resolves it (in a subshell: the helper also checks GCC)
CUDA_FULL_VERSION="$(bash -c 'source scripts/cuda_helper.sh >/dev/null 2>&1; echo "${CUDA_FULL_VERSION}"')"
[[ -n ${CUDA_FULL_VERSION} ]] || { echo "upstream doesn't know CUDA_VERSION=${CUDA_VERSION}" >&2; exit 1; }
toolkit="${GRUT}/.venv/cuda-${CUDA_FULL_VERSION}"

assemble_toolkit() {   # assemble_toolkit <dest>: CUDA from redist tarballs for this architecture
  local dest=$1 plat tmp
  case "${ARCH}" in x86_64) plat=linux-x86_64 ;; aarch64) plat=linux-sbsa ;; *) echo "unsupported arch ${ARCH}" >&2; exit 1 ;; esac
  local base=https://developer.download.nvidia.com/compute/cuda/redist
  tmp="$(mktemp -d)"
  curl -fsSL "${base}/redistrib_${CUDA_FULL_VERSION}.json" -o "${tmp}/manifest.json"
  python3 - "${tmp}/manifest.json" "${plat}" > "${tmp}/files" <<'PY'
import json, sys
m, plat = json.load(open(sys.argv[1])), sys.argv[2]
want = ("cuda_cccl cuda_crt cuda_cudart cuda_culibos cuda_cupti cuda_cuobjdump cuda_cuxxfilt cuda_nvcc cuda_nvdisasm "
        "cuda_nvml_dev cuda_nvprune cuda_nvrtc cuda_nvtx cuda_profiler_api libcublas libcufft libcurand libcusolver "
        "libcusparse libnvfatbin libnvjitlink libnvptxcompiler libnvvm").split()
for k in want:
    if k in m and plat in m[k]:
        print(m[k][plat]["relative_path"])
PY
  echo "assembling CUDA ${CUDA_FULL_VERSION} (${plat}) from $(wc -l < "${tmp}/files") redist components into ${dest}"
  mkdir -p "${dest}"
  while read -r rel; do
    curl -fsSL "${base}/${rel}" | tar -xJ -C "${tmp}"
    cp -a "${tmp}/$(basename "${rel}" .tar.xz)/." "${dest}/"
    rm -rf "${tmp:?}/$(basename "${rel}" .tar.xz)"
  done < "${tmp}/files"
  rm -rf "${tmp}"
  [[ -e ${dest}/lib64 ]] || ln -s lib "${dest}/lib64"   # the runfile layout, which torch's extension builder expects
  "${dest}/bin/nvcc" --version | tail -1
}

[[ -d .venv ]] || uv venv .venv --python 3.11 --prompt 3dgrut
if [[ ! -x ${toolkit}/bin/nvcc && ${ARCH} != x86_64 ]]; then
  assemble_toolkit "${toolkit}"   # upstream only knows the x86_64 runfiles
fi
FORCE_LOCAL_CUDA=1 ./scripts/create_venv_cuda.sh 3dgrut
# shellcheck source=/dev/null
source .venv/bin/activate
MAX_JOBS="${MAX_JOBS:-${JOBS}}" ./install_env_uv.sh 3dgrut

if ! python -c "import pxr" 2>/dev/null; then
  echo "no usd-core wheel for ${ARCH}: building OpenUSD ${USD_TAG} into .venv/opt/usd"
  build="${GRUT}/.venv/opt/usd-build"
  [[ -d ${build}/OpenUSD ]] || git clone -q --depth 1 --branch "${USD_TAG}" https://github.com/PixarAnimationStudios/OpenUSD.git "${build}/OpenUSD"
  [[ -x ${build}/tools/bin/cmake ]] || { uv venv -q "${build}/tools" && uv pip install -q -p "${build}/tools" cmake; }
  PATH="${build}/tools/bin:${PATH}" python "${build}/OpenUSD/build_scripts/build_usd.py" \
    --no-imaging --no-materialx --no-examples --no-tutorials --no-tools --no-docs --no-tests -j "${JOBS}" \
    --build "${build}/build" --src "${build}/src" "${GRUT}/.venv/opt/usd" > "${build}.log" 2>&1 \
    || { tail -30 "${build}.log"; exit 1; }
  site="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  ls -d "${GRUT}"/.venv/opt/usd/lib/python3*/site-packages > "${site}/openusd.pth"
  rm -rf "${build}/build" "${build}/src"
fi
python -c "import torch, kaolin, tinycudann; from pxr import Usd; print('3dgrut ready: torch', torch.__version__, 'cuda', torch.version.cuda, '| usd', '.'.join(map(str, Usd.GetVersion())))"
