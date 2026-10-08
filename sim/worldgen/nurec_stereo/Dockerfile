# NuRec stereo-workflow toolchain in one image, so nothing is installed on the host:
#   isaac_mapping_ros  rosbag -> images + frames_meta (rosbag_to_mapping_data), FoundationStereo offline depth
#   pyCuSFM            poses + sparse map (cusfm_cli); binaries come from Git LFS, fetched here
#   nvblox             TSDF mesh from depth + poses (fuse_cusfm)
# 3DGRUT training then runs with sim/nurec/3dgrut on the host (already installed there).
# Base: NVIDIA's Isaac ROS 5.0 dev image (ROS 2 Lyrical, CUDA 13, TensorRT), pulled anonymously from NGC.
# The bags come from Jazzy; mcap + standard message types read fine under Lyrical.
ARG ISAAC_ROS_IMAGE=nvcr.io/nvidia/isaac/ros:isaac_ros_1742a70509cba44cb1943ab200bbece1-amd64
FROM ${ISAAC_ROS_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
# The base image is Isaac ROS 5.0 on ROS 2 Lyrical, with its apt repository (release-5.0) already configured.
# Packages must come from that release: mixing in release-4 (Jazzy) breaks its pinned Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
      git git-lfs cmake build-essential \
      libopencv-dev libgoogle-glog-dev libgflags-dev libabsl-dev libprotobuf-dev protobuf-compiler \
      libsqlite3-dev libbenchmark-dev libgtest-dev \
      ros-lyrical-isaac-mapping-ros \
      ros-lyrical-isaac-ros-foundationstereo-models-install \
      ros-lyrical-rosbag2-storage-mcap ros-lyrical-rosbag2-compression-zstd \
    && rm -rf /var/lib/apt/lists/*

# FoundationStereo TensorRT models are installed at run time (run.sh setup): building the engine needs the GPU,
# which docker build doesn't get. Downloading them means accepting NVIDIA's model EULA; ISAAC_ROS_ACCEPT_EULA=1 is
# the helper's non-interactive acceptance, agreed by the repo owner on 2026-10-08 when choosing this workflow.
# Models land under $ISAAC_ROS_WS/isaac_ros_assets (a mounted cache, see run.sh).
ENV ISAAC_ROS_WS=/opt/isaac_ros_ws ISAAC_ROS_ACCEPT_EULA=1

# pyCuSFM (Apache-2.0); prebuilt CUDA 13 binaries are LFS objects.
RUN git lfs install --system \
    && git clone --depth 1 https://github.com/nvidia-isaac/pyCuSFM.git /opt/pyCuSFM \
    && cd /opt/pyCuSFM && git lfs pull \
    && ./setup.bash cuda13 \
    && pip3 install --break-system-packages -e .

# nvblox core library + executables (fuse_cusfm). No GPU during docker build, so the CUDA arch is given:
# 120 = Blackwell (RTX 5090). Add more (e.g. "89;120") for other GPUs.
RUN git clone --branch public --depth 1 https://github.com/nvidia-isaac/nvblox.git /opt/nvblox \
    && cmake -S /opt/nvblox -B /opt/nvblox/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
         -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_PYTORCH_WRAPPER=OFF -DBUILD_RENDERER=OFF \
    && cmake --build /opt/nvblox/build -j"$(nproc)"

ENV PATH=/opt/nvblox/build/executables:${PATH}
WORKDIR /workspace
