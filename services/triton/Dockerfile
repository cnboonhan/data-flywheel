# Triton Inference Server plus what the Python-backend models (XPolicyLab ACT) and the sync-triton job need:
# torch/torchvision for CUDA 13 (GB300 is sm_103; the image runs CUDA 13.x in forward-compatibility mode on driver 580),
# onnx + onnxscript for exporting, mlflow-skinny and boto3 for reading the registry and writing the triton bucket.
FROM nvcr.io/nvidia/tritonserver:26.08-py3
RUN pip install --no-cache-dir --break-system-packages \
      --index-url https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 torchvision==0.29.1+cu130 \
 && pip install --no-cache-dir --break-system-packages \
      onnx onnxscript "mlflow-skinny==3.17.0" boto3 h5py pyyaml tritonclient[http]
