# Triton Inference Server with the vLLM backend (LLMs and VLMs; the image's torch is built for sm_103, GB300), plus the
# ONNX Runtime backend from the standard image (MLP), and what the Python-backend models (XPolicyLab ACT) and the
# sync-triton job need: onnx + onnxscript for exporting, mlflow-skinny and boto3 for reading the registry and writing
# the triton bucket (these only add packages; none of the image's own, vLLM's included, change).
FROM nvcr.io/nvidia/tritonserver:26.08-py3 AS standard
FROM nvcr.io/nvidia/tritonserver:26.08-vllm-python-py3
COPY --from=standard /opt/tritonserver/backends/onnxruntime /opt/tritonserver/backends/onnxruntime
RUN pip install --no-cache-dir --break-system-packages \
      onnx onnxscript "mlflow-skinny==3.17.0" boto3 h5py pyyaml tritonclient[http]
