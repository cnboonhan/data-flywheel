# Triton Inference Server with the vLLM backend (LLMs and VLMs; the image's torch is built for sm_103, GB300), plus the
# ONNX Runtime backend from the standard image (MLP), and what the Python-backend models and the sync-triton job need:
# onnx + onnxscript for exporting, mlflow-skinny and boto3 for reading the registry and writing the triton bucket,
# diffusers (Qwen-Image-Edit). Everything is installed against a constraints file of the image's own torch, numpy,
# transformers and vLLM, so none of them changes. Pins: mlflow-skinny is the MLflow server's version
# (services/mlflow/Dockerfile); diffusers>=0.36 is the floor Qwen-Image-Edit-2511 was run on (its card asks for the latest).
#
# π0.5 (XPolicyLab's policy/Pi_05: openpi, JAX) gets a venv of its own, /opt/venv-pi05: openpi pins its own JAX, torch
# and numpy, which the image's can't share. It is built like this:
# openpi's uv.lock exported (with the lerobot group) and installed as is, minus torchcodec (no Linux arm64 wheels), then
# openpi and openpi-client themselves. The openpi source is the build context `openpi` (compose.yml). The backend
# (a π0.5 model's 1/model.py) runs each checkpoint in it as a child process, without the image's dist-packages on
# LD_LIBRARY_PATH (else the venv's torch loads the image's libtorch, built for Python 3.12).
FROM nvcr.io/nvidia/tritonserver:26.08-py3 AS standard
FROM nvcr.io/nvidia/tritonserver:26.08-vllm-python-py3
COPY --from=standard /opt/tritonserver/backends/onnxruntime /opt/tritonserver/backends/onnxruntime
RUN pip list --format=freeze | grep -iE '^(torch|torchvision|numpy|transformers|vllm|triton|tokenizers|huggingface.hub|safetensors)==' > /tmp/pinned.txt \
 && cat /tmp/pinned.txt \
 && pip install --no-cache-dir --break-system-packages -c /tmp/pinned.txt \
      onnx onnxscript "mlflow-skinny==3.17.0" boto3 h5py pyyaml tritonclient[http] \
      "diffusers>=0.36" accelerate opencv-python-headless scipy tqdm \
 && pip list --format=freeze | grep -iE '^(torch|torchvision|numpy|transformers|vllm|triton|tokenizers|huggingface.hub|safetensors)==' | diff - /tmp/pinned.txt
COPY --from=openpi . /opt/openpi
# uv's own Python 3.11 under /opt (its default, root's home, isn't readable by the service account Triton runs as)
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN pip install --no-cache-dir --break-system-packages uv==0.12.8 \
 && uv venv --python 3.11 /opt/venv-pi05 \
 && (cd /opt/openpi && uv export --frozen --no-dev --group lerobot --no-emit-project --no-emit-workspace --no-hashes -o /tmp/pi05.txt) \
 && grep -v -E '^torchcodec==' /tmp/pi05.txt > /tmp/pi05-arm64.txt \
 && uv pip install --python /opt/venv-pi05/bin/python --no-cache --no-deps -r /tmp/pi05-arm64.txt --index-strategy unsafe-best-match \
 && uv pip install --python /opt/venv-pi05/bin/python --no-cache --no-deps /opt/openpi/packages/openpi-client /opt/openpi \
 && uv pip install --python /opt/venv-pi05/bin/python --no-cache pytest \
 && rm -rf /opt/openpi/build /opt/openpi/packages/openpi-client/build \
 && chmod -R a+rX /opt/venv-pi05 /opt/uv-python \
 && LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v dist-packages | paste -sd:) \
    /opt/venv-pi05/bin/python -c 'import openpi.models.model, jax; print("openpi ok, jax", jax.__version__)'
