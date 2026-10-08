#!/usr/bin/env bash
#SBATCH --job-name=install-robodojo
#SBATCH --partition=raus_manual
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.log
#
# Build the RoboDojo evaluation environment (Isaac Sim 5.1, IsaacLab 2.3,
# curobo) as a uv venv, following RoboDojo/scripts/install.sh step by step.
# That script assumes conda and x86_64 Miniconda; this one uses uv and the
# aarch64 wheels (Isaac Sim and torch cu128 publish them). Also installs the
# conda shim the eval scripts need (they call `conda activate <env>`).
#   sbatch install-robodojo.sbatch
set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/workspaces/data-flywheel/eval/system1/RoboDojo}
ENVS_DIR=${ENVS_DIR:-/tier1/htx_boonhan/services/envs}
PIPELINES_ROOT=${PIPELINES_ROOT:-/tier1/htx_boonhan/services/pipelines}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tier1/htx_boonhan/services/uv-cache}
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES TERM=xterm-256color

pins=(numpy==1.26.0 packaging==23.0 typing_extensions==4.12.2 filelock==3.13.1 websockets==12.0 click==8.1.7 psutil==5.9.8
      wheel==0.45.1 starlette==0.45.3 scipy==1.15.3 warp-lang==1.11.0 "onnx>=1.18,<1.22" "ipython<9" virtualenv==20.30.0)
torch_stack=(torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128)

mkdir -p "$ENVS_DIR"
[[ -d $ENVS_DIR/robodojo ]] || uv venv --python 3.11 "$ENVS_DIR/robodojo"
source "$ENVS_DIR/robodojo/bin/activate"
cd "$PROJECT_ROOT"

echo "[0/6] system libraries the node lacks, in user space"
# Isaac Sim's RTX/MDL stack dlopens libGLU, libOpenGL (glvnd) and a few X11 helpers that aren't
# installed on the compute nodes. Without root, take them from the Ubuntu arm64 packages and put
# them on LD_LIBRARY_PATH (evaluate-xpolicylab.sbatch does that).
ROBODOJO_DIR=${ROBODOJO_DIR:-/tier1/htx_boonhan/services/robodojo}
mkdir -p "$ROBODOJO_DIR/lib" "$ROBODOJO_DIR/deb" && cd "$ROBODOJO_DIR/deb"
for p in libglu1-mesa libopengl0 libglvnd0 libglx0 libgl1 libegl1 libxt6 libxmu6 libxi6; do
  ls "$p"_*.deb >/dev/null 2>&1 || apt-get download "$p" >/dev/null 2>&1 || echo "could not download $p"
done
for d in *.deb; do dpkg-deb -x "$d" extracted; done
cp -a extracted/usr/lib/aarch64-linux-gnu/*.so* "$ROBODOJO_DIR/lib/" 2>/dev/null || true
cd "$PROJECT_ROOT"

echo "[1/6] base deps"
uv pip install -q huggingface_hub hf_transfer transforms3d msgpack-numpy pyyaml
uv pip install -q open3d || echo "open3d unavailable on this platform; skipping (only used by tooling)"
uv pip install -q opencv-python-headless==4.11.0.86 pillow matplotlib scipy==1.15.3 scikit-learn numpy==1.26.0 setuptools

echo "[2/6] torch cu128 + Isaac Sim 5.1"
uv pip install -q numpy==1.26.0 typing_extensions==4.12.2 filelock==3.13.1
uv pip install -q "${torch_stack[@]}"
uv pip install -q "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
uv pip install -q "${pins[@]}"

echo "[3/6] IsaacLab"
# isaaclab.sh's ARM branch installs torch 2.9/cu130 (its tested stack for aarch64) and exits
# non-zero after printing an LD_PRELOAD notice; RoboDojo would re-pin 2.7/cu128 on x86. On
# aarch64 keep IsaacLab's choice: the driver (580) supports CUDA 13 and the cu128 aarch64
# wheels don't come with torchaudio.
# Editable installs must point at this checkout (re-run after moving the repo): skip only if they already do.
if ! python -c "import isaaclab, sys; sys.exit(0 if isaaclab.__file__.startswith('$PROJECT_ROOT/') else 1)" 2>/dev/null; then
  (cd third_party/IsaacLab && ./isaaclab.sh --install none) || echo "isaaclab.sh exited non-zero; checking the install below"
fi
[[ $(uname -m) == aarch64 ]] || uv pip install -q "${torch_stack[@]}"
uv pip install -q "${pins[@]}"
python -c "import isaaclab, torch; print('isaaclab ok, torch', torch.__version__)"

echo "[4/6] curobo"
uv pip install -q scikit-build-core cmake ninja setuptools wheel   # build deps for --no-build-isolation (xatlas needs scikit-build-core)
cu=$(python -c "import torch; print('cu13' if torch.version.cuda.startswith('13') else 'cu12')")
constraints=$(mktemp); printf '%s\n' "${pins[@]}" "stable-baselines3<2.8" > "$constraints"
reinstall=(); python -c "import curobo, sys; sys.exit(0 if curobo.__file__.startswith('$PROJECT_ROOT/') else 1)" 2>/dev/null || reinstall=(--reinstall-package nvidia-curobo)
(cd third_party/curobo && uv pip install -q -e ".[$cu]" --no-build-isolation --constraint "$constraints" "${reinstall[@]}")
uv pip install -q "${pins[@]}"
uv pip uninstall -q python-discovery 2>/dev/null || true

echo "[5/6] XPolicyLab (client side) + conda shim + ffmpeg"
uv pip install -q -e XPolicyLab --reinstall-package xpolicylab
# The nodes have no ffmpeg; RoboDojo's eval client pipes camera frames through one. imageio-ffmpeg ships a static build.
uv pip install -q imageio-ffmpeg
mkdir -p "$ROBODOJO_DIR/bin" && ln -sf "$(ls "$ENVS_DIR"/robodojo/lib/python3.*/site-packages/imageio_ffmpeg/binaries/ffmpeg-* | head -1)" "$ROBODOJO_DIR/bin/ffmpeg"
mkdir -p "$ENVS_DIR/conda-shim"
cp -r "$PIPELINES_ROOT/slurm/conda-shim/." "$ENVS_DIR/conda-shim/"
chmod +x "$ENVS_DIR/conda-shim/bin/conda"

echo "[6/6] check"
# On aarch64 Isaac Sim's libs need libgomp preloaded (isaaclab.sh prints this; the eval job exports it too).
[[ $(uname -m) == aarch64 ]] && export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/lib/aarch64-linux-gnu/libgomp.so.1"
export LD_LIBRARY_PATH="$ROBODOJO_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python - <<EOF
import torch, isaacsim, isaaclab, curobo, XPolicyLab
print("torch", torch.__version__, "cuda build", torch.version.cuda, "| isaacsim ok | isaaclab", getattr(isaaclab, "__version__", "?"), "| curobo ok")
for m in (isaaclab, curobo, XPolicyLab):
    assert m.__file__.startswith("$PROJECT_ROOT/"), f"{m.__name__} is installed from {m.__file__}, not from $PROJECT_ROOT"
print("editable installs point at $PROJECT_ROOT")
EOF
echo "robodojo env ready at $ENVS_DIR/robodojo"
