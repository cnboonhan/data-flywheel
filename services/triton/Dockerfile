# Triton Inference Server with the vLLM backend (LLMs and VLMs; the image's torch is built for sm_103, GB300), plus the
# ONNX Runtime backend from the standard image (MLP), and what the Python-backend models and the sync-triton job need:
# onnx + onnxscript for exporting, mlflow-skinny and boto3 for reading the registry and writing the triton bucket,
# diffusers (Qwen-Image-Edit) and MoGe with its git dependencies (MoGe-3; FlexGEMM runs on Triton kernels, no build).
# Everything is installed against a constraints file of the image's own torch, numpy, transformers and vLLM, so none of
# them changes; MoGe and its git dependencies go in without theirs (they would pull numpy 2 and an OpenGL renderer).
FROM nvcr.io/nvidia/tritonserver:26.08-py3 AS standard
FROM nvcr.io/nvidia/tritonserver:26.08-vllm-python-py3
COPY --from=standard /opt/tritonserver/backends/onnxruntime /opt/tritonserver/backends/onnxruntime
RUN pip list --format=freeze | grep -iE '^(torch|torchvision|numpy|transformers|vllm|triton|tokenizers|huggingface.hub|safetensors)==' > /tmp/pinned.txt \
 && cat /tmp/pinned.txt \
 && pip install --no-cache-dir --break-system-packages -c /tmp/pinned.txt \
      onnx onnxscript "mlflow-skinny==3.17.0" boto3 h5py pyyaml tritonclient[http] \
      "diffusers>=0.36" accelerate opencv-python-headless scipy tqdm \
 && pip install --no-cache-dir --break-system-packages --no-deps \
      "moge @ git+https://github.com/microsoft/MoGe.git@74fbce054ebe" \
      "utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
      "pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4" \
      "flex-gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf" \
 && pip list --format=freeze | grep -iE '^(torch|torchvision|numpy|transformers|vllm|triton|tokenizers|huggingface.hub|safetensors)==' | diff - /tmp/pinned.txt
