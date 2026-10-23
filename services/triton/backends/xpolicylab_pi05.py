"""Triton Python backend for a π0.5 checkpoint (XPolicyLab's policy/Pi_05: openpi, JAX), copied in as
<model>/<version>/model.py by sync.py with serve_pi05.py and r1pro.py (sensors/autolearn/adapter/pi05/: the HTTP
server and the openpi configs it registers).

openpi runs in the image's own venv (/opt/venv-pi05, Dockerfile), not Triton's Python, so each instance starts
serve_pi05.py there as a child process on its GPU and forwards requests to it on 127.0.0.1. Stateless: one request =
one observation, the response is the whole action chunk.

    inputs   state            FP32  [D]          the joint vector the checkpoint was trained on (r1pro: 20 values)
             cam_high         UINT8 [H, W, 3]    RGB, any size (openpi resizes)
             cam_left_wrist   UINT8 [H, W, 3]
             cam_right_wrist  UINT8 [H, W, 3]
             prompt           BYTES [1]          optional, default the run's `prompt` param
    output   action           FP32  [H, D]       the predicted chunk, absolute joint targets
Parameters (sync.py): weights (the checkpoint step directory, in place under /mlflow), openpi_config, prompt.
"""

import io
import json
import os
import socket
import subprocess
import time
import urllib.request

import numpy as np
import triton_python_backend_utils as pb_utils

START_S = 900   # wait for the child to load the checkpoint: ~1-2 min on the cluster, more when JAX compiles; untuned


class TritonPythonModel:
    def initialize(self, args):
        p = {k: v["string_value"] for k, v in json.loads(args["model_config"])["parameters"].items()}
        here = os.path.dirname(os.path.abspath(__file__))
        self.prompt = p.get("prompt", "")
        with socket.socket() as s:   # a free port on 127.0.0.1 for this instance's child
            s.bind(("127.0.0.1", 0))
            self.url = f"http://127.0.0.1:{s.getsockname()[1]}"
        # LD_LIBRARY_PATH without the image's dist-packages: the venv's torch would load the image's libtorch (Python 3.12)
        lib = ":".join(d for d in os.environ.get("LD_LIBRARY_PATH", "").split(":") if d and "dist-packages" not in d)
        env = {**os.environ, "LD_LIBRARY_PATH": lib, "CUDA_VISIBLE_DEVICES": str(args.get("model_instance_device_id", "0")), "HOME": "/tmp",
               "PYTHONPATH": here, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",   # share the GPU: allocate as it goes
               # jaxlib does not know the GB300's compute capability 10.3 (train_pi05.sbatch): Triton GEMMs off
               "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"}
        self.child = subprocess.Popen(["/opt/venv-pi05/bin/python", os.path.join(here, "serve_pi05.py"), "--ckpt",
                                       p["weights"], "--config", p["openpi_config"], "--port", self.url.rsplit(":", 1)[1]],
                                      env=env)
        end = time.time() + START_S
        while time.time() < end:
            if self.child.poll() is not None:
                raise RuntimeError(f"serve_pi05.py exited with {self.child.returncode} (Triton's log has its output)")
            try:
                urllib.request.urlopen(self.url + "/", timeout=5)
                return
            except OSError:
                time.sleep(5)
        raise RuntimeError(f"serve_pi05.py did not answer in {START_S} s")

    def execute(self, requests):
        out = []
        for req in requests:
            get = lambda n: pb_utils.get_input_tensor_by_name(req, n)
            prompt = get("prompt")
            obs = {"state": get("state").as_numpy().astype(np.float32), "head_rgb": get("cam_high").as_numpy(),
                   "left_wrist_rgb": get("cam_left_wrist").as_numpy(), "right_wrist_rgb": get("cam_right_wrist").as_numpy(),
                   "task": np.asarray(prompt.as_numpy().reshape(-1)[0].decode() if prompt is not None else self.prompt)}
            try:
                buf = io.BytesIO()
                np.savez(buf, **obs)
                body = urllib.request.urlopen(urllib.request.Request(self.url + "/act", buf.getvalue()), timeout=120).read()
                action = np.load(io.BytesIO(body))["action"].astype(np.float32)
                out.append(pb_utils.InferenceResponse([pb_utils.Tensor("action", action)]))
            except Exception as e:   # one bad request fails alone
                out.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(f"{type(e).__name__}: {e}")))
        return out

    def finalize(self):
        self.child.terminate()
