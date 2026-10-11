"""A client for one model served by Triton, its endpoint set per role, so each tool can point each model it needs at
any Triton server (README.md, "Call a model from your own code").

    from triton_endpoint import Endpoint
    edit = Endpoint("EDIT", "Qwen-Image-Edit-2511")       # EDIT_ENDPOINT overrides
    out = edit.infer({"images": [png], "prompt": [b"..."]}, ["image"])

<ROLE>_ENDPOINT is either a full URL, https://<host:port>/v2/models/<model> (any Triton server), or a model name on
the flywheel Triton, or unset: the default model there. The flywheel Triton is TRITON_URL, else
triton.$SERVICE_HOST:$CADDY_PORT (quickstart.md). The token is <ROLE>_TOKEN, else TRITON_TOKEN; the CA
bundle <ROLE>_CA, else AWS_CA_BUNDLE. Needs numpy and tritonclient[http].
"""

import json
import os
import ssl
import threading
import urllib.parse
import urllib.request

import numpy as np


class Endpoint:
    def __init__(self, role: str, default_model: str):
        spec = os.environ.get(f"{role}_ENDPOINT") or default_model
        if "://" in spec:
            u = urllib.parse.urlparse(spec)
            parts = u.path.strip("/").split("/")
            if len(parts) < 3 or parts[:2] != ["v2", "models"]:
                raise SystemExit(f"{role}_ENDPOINT={spec}: expected <scheme>://<host:port>/v2/models/<model>")
            self.scheme, self.host, self.model = u.scheme, u.netloc, parts[2]
        else:
            host, port = os.environ.get("SERVICE_HOST"), os.environ.get("CADDY_PORT")
            self.scheme, self.model = "https", spec
            self.host = os.environ.get("TRITON_URL") or (host and port and f"triton.{host}:{port}")
            if not self.host:
                raise SystemExit(f"{role}: set {role}_ENDPOINT, or TRITON_URL (or SERVICE_HOST and CADDY_PORT): "
                                 "see quickstart.md")
        self.role = role
        self.token = os.environ.get(f"{role}_TOKEN") or os.environ.get("TRITON_TOKEN")
        self.ca = os.environ.get(f"{role}_CA") or os.environ.get("AWS_CA_BUNDLE")
        self._local = threading.local()                       # one connection per thread

    def __repr__(self):
        return f"{self.role}: {self.scheme}://{self.host}/v2/models/{self.model}"

    @property
    def headers(self) -> dict:
        return {"Authorization": "Bearer " + self.token} if self.token else {}

    def _client(self):
        if not hasattr(self._local, "client"):
            import tritonclient.http as th
            tls = self.scheme == "https"
            # 15 min to answer (the slowest call, a MoGe panorama, takes ~2 min; README.md), 1 min to connect; untuned.
            self._local.client = th.InferenceServerClient(
                self.host, ssl=tls, ssl_options={"ca_certs": self.ca} if tls and self.ca else None,
                network_timeout=900.0, connection_timeout=60.0)
        return self._local.client

    def infer(self, inputs: dict, outputs: list[str]) -> dict:
        """inputs: name -> numpy array, or a list of bytes for a BYTES tensor; returns name -> numpy array."""
        import tritonclient.http as th
        from tritonclient.utils import np_to_triton_dtype
        tensors = []
        for name, a in inputs.items():
            a = np.array(a, dtype=object) if isinstance(a, list) else np.asarray(a)
            t = th.InferInput(name, list(a.shape), "BYTES" if a.dtype == object else np_to_triton_dtype(a.dtype))
            t.set_data_from_numpy(a, binary_data=True)
            tensors.append(t)
        r = self._client().infer(self.model, tensors, headers=dict(self.headers),   # tritonclient adds to the dict
                                 outputs=[th.InferRequestedOutput(o, binary_data=True) for o in outputs])
        return {o: r.as_numpy(o) for o in outputs}

    def generate(self, body: dict, timeout: float = 900) -> dict:   # s, as infer's network_timeout
        """POST to the model's generate endpoint (LLMs and VLMs on the vLLM backend)."""
        req = urllib.request.Request(f"{self.scheme}://{self.host}/v2/models/{self.model}/generate",
                                     json.dumps(body).encode(), {"Content-Type": "application/json", **self.headers})
        ctx = ssl.create_default_context(cafile=self.ca) if self.scheme == "https" else None
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return json.loads(r.read())
