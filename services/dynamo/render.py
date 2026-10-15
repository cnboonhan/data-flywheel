"""Render models.yaml into the compose files of Dynamo's two trays and the gateway's model list.

    python render.py OUT_DIR      (ctl.sh dynamo render: OUT_DIR = $STATE_DIR/dynamo)

Writes head.json (DYNAMO_HEAD: etcd, a frontend per model, the gateway, each model's rank-0 engine), worker.json
(DYNAMO_WORKER: each model's headless rank-1 engine) and gateway.json. Everything runs on the host network; all but
the gateway and the vLLM rendezvous listen on 127.0.0.1, since Dynamo has no authentication of its own.

Each model's weights are the MLflow version its registered model's alias `dynamo` points to, read in place from the
mlflow bucket; that version's tag `dynamo.args` (a JSON list) appends engine flags. Env: STATE_DIR, SERVICE_UID,
SERVICE_GID, DYNAMO_HEAD_ADDR, MLFLOW_TRACKING_URI/USERNAME/PASSWORD (+ MLFLOW_TRACKING_SERVER_CERT_PATH).
Engines don't restart by themselves (restart: "no"): all at once they would not fit the GPUs; `ctl.sh dynamo up`
starts them one at a time and puts all but one to sleep.
"""
import base64
import json
import os
import ssl
import sys
import urllib.parse
import urllib.request

import yaml

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
cfg = yaml.safe_load(open(os.path.join(here, "models.yaml")))
state, head = os.environ["STATE_DIR"], os.environ["DYNAMO_HEAD_ADDR"]
user = f'{os.environ["SERVICE_UID"]}:{os.environ["SERVICE_GID"]}'
GATEWAY_PORT = 8199   # the gateway on the head's address, which Caddy's llm.<host> reaches; any free port


def mlflow(path, **query):
    url = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/api/2.0/mlflow/" + path + "?" + urllib.parse.urlencode(query)
    auth = base64.b64encode(f'{os.environ["MLFLOW_TRACKING_USERNAME"]}:{os.environ["MLFLOW_TRACKING_PASSWORD"]}'.encode())
    ctx = ssl.create_default_context(cafile=os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH"))
    req = urllib.request.Request(url, headers={"Authorization": "Basic " + auth.decode()})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        return json.load(r)


def served_version(registered):
    """(path of the weights inside the engine container, extra flags) of the version aliased `dynamo`, or None."""
    try:
        mv = mlflow("registered-models/alias", name=registered, alias="dynamo")["model_version"]
    except urllib.error.HTTPError as e:
        if e.code == 404 or (e.code == 400 and "alias" in e.read().decode()):   # no such model; model but no alias
            return None
        raise
    tags = {t["key"]: t["value"] for t in mv.get("tags", [])}
    src = mv["source"]
    if not src.startswith("mlflow-artifacts:/"):
        raise SystemExit(f"{registered} v{mv['version']}: source {src} is not in the mlflow bucket")
    return "/mlflow/" + src.removeprefix("mlflow-artifacts:/").lstrip("/"), json.loads(tags.get("dynamo.args", "[]"))


common = dict(network_mode="host", user=user, ipc="host", shm_size="32gb",
              logging={"driver": "json-file", "options": {"max-size": "20m", "max-file": "5"}})   # 100 MB of logs; untuned


def namespace(m):
    """Each model's own Dynamo namespace: in a shared one, every engine registers at the same endpoint and the second fails."""
    return "flywheel-" + m["name"].replace(".", "-")


def engine(m, path, extra, rank):
    p = m["ports"]
    cmd = ["numactl", "--membind", cfg["host_numa_nodes"],   # host memory on the CPU nodes only (models.yaml)
           "python3", "-m", "dynamo.vllm", "--model", path, "--served-model-name", m["name"],
           *cfg["common_args"], *m["args"], *extra,
           "--nnodes", "2", "--node-rank", str(rank), "--master-addr", head, "--master-port", str(p["rendezvous"])]
    if rank:
        cmd.append("--headless")   # rank 1 only computes; no discovery, no system API
    env = {**cfg["common_env"], **m.get("env", {}), "HOME": "/cache", "HF_HOME": "/cache/hf",
           "ETCD_ENDPOINTS": "http://127.0.0.1:2379", "DYN_REQUEST_PLANE": "tcp", "DYN_EVENT_PLANE": "zmq",
           "DYN_TCP_RPC_HOST": "127.0.0.1", "DYN_SYSTEM_HOST": "127.0.0.1", "DYN_SYSTEM_PORT": str(p["system"]),
           "DYN_SYSTEM_ENABLED": "true", "DYN_NAMESPACE": namespace(m)}
    return {**common, "image": m["image"], "restart": "no", "command": cmd, "environment": env,
            "devices": ["nvidia.com/gpu=all", "/dev/infiniband"], "cap_add": ["IPC_LOCK", "SYS_NICE"],   # IB; numactl
            "ulimits": {"memlock": -1, "stack": 67108864},
            "volumes": [f"{state}/versitygw/buckets/mlflow:/mlflow:ro", f"{state}/dynamo/cache/{m['name']}:/cache"]}


head_svc = {"etcd": {**common, "image": "bitnamilegacy/etcd:3.6.1", "restart": "unless-stopped",
                     "environment": {"ALLOW_NONE_AUTHENTICATION": "yes",
                                     "ETCD_LISTEN_CLIENT_URLS": "http://127.0.0.1:2379",
                                     "ETCD_ADVERTISE_CLIENT_URLS": "http://127.0.0.1:2379",
                                     "ETCD_LISTEN_PEER_URLS": "http://127.0.0.1:2380"},
                     "volumes": [f"{state}/dynamo/etcd:/bitnami/etcd"]}}   # the image's data path; ETCD_DATA_DIR isn't honoured
worker_svc, gw_models, skipped = {}, {}, []
for m in cfg["models"]:
    v = served_version(m["registered"])
    if v is None:
        skipped.append(m["registered"])
        continue
    path, extra = v
    p, n = m["ports"], m["name"]
    head_svc[f"frontend-{n}"] = {**common, "image": m["image"], "restart": "unless-stopped",
                                 "command": ["python3", "-m", "dynamo.frontend", "--http-host", "127.0.0.1",
                                             "--http-port", str(p["frontend"]), "--router-mode", "round-robin",
                                             "--namespace", namespace(m),
                                             *m.get("frontend_args", [])],
                                 "environment": {"ETCD_ENDPOINTS": "http://127.0.0.1:2379", "DYN_REQUEST_PLANE": "tcp",
                                                 "DYN_EVENT_PLANE": "zmq", "HOME": "/tmp"},
                                 "depends_on": ["etcd"]}
    head_svc[f"engine-{n}"] = {**engine(m, path, extra, 0), "depends_on": ["etcd"]}
    worker_svc[f"engine-{n}"] = engine(m, path, extra, 1)
    gw_models[n] = {"frontend": f"http://127.0.0.1:{p['frontend']}", "system": f"http://127.0.0.1:{p['system']}"}
head_svc["gateway"] = {**common, "build": os.path.join(here, "..", "llm-gateway"), "image": "flywheel-llm-gateway:latest",
                       "restart": "unless-stopped", "user": user,
                       "environment": {"GATEWAY_CONFIG": "/config/gateway.json", "GATEWAY_LISTEN": f"{head}:{GATEWAY_PORT}",
                                       "LLM_TOKEN": "${LLM_TOKEN}", "LLM_CRITICAL_TOKEN": "${LLM_CRITICAL_TOKEN}"},
                       "volumes": [f"{out}:/config:ro"]}
os.makedirs(out, exist_ok=True)
json.dump({"name": "flywheel-dynamo", "services": head_svc}, open(os.path.join(out, "head.json"), "w"), indent=1)
json.dump({"name": "flywheel-dynamo", "services": worker_svc}, open(os.path.join(out, "worker.json"), "w"), indent=1)
json.dump({"models": gw_models, "default": cfg["default_awake"] if cfg["default_awake"] in gw_models else next(iter(gw_models), None),
           "order": [n for n in (m["name"] for m in cfg["models"]) if n in gw_models]},
          open(os.path.join(out, "gateway.json"), "w"), indent=1)
print(f"rendered {', '.join(gw_models) or 'no models'} into {out}" + (f"; no `dynamo` alias yet: {', '.join(skipped)}" if skipped else ""))
