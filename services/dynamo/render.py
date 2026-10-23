"""Render the deployment spec (models.yaml) into one compose file per node, the gateways' model lists and the
Prometheus targets.

    python render.py OUT_DIR [SPEC]   (ctl.sh dynamo render: OUT_DIR = $STATE_DIR/dynamo, SPEC = the Gitea repo
                                       dynamo's models.yaml; default: the models.yaml beside this file, the seed)

The spec lives in the Gitea repo dynamo, so deploying is an edit there. A model is served when its entry has a
`version`: that version of its MLflow registered model (the admin registers the weights, download-models-hf), read in
place from the mlflow bucket. Its placement says where: `trays` (both of DYNAMO_HEAD's and DYNAMO_WORKER's 8 GPUs, rank
0 on the head and rank 1 headless on the worker; one such model at a time) or a node and GPUs with a share of each GPU's
memory (the shares on one GPU must fit its budget in the spec's `gpus`). A changed spec is served at the next sync
(Gitea workflow sync-dynamo, which a push of the spec starts, or `ctl.sh dynamo up`). The spec is checked first
(check_spec): whoever can push it chooses what runs on the GPU nodes.

Writes OUT_DIR/<node>.json (each node's services: etcd, the frontends and engines of the models whose rank 0 runs there,
and a gateway; rank-1 engines on the worker), OUT_DIR/<node>.gateway.json (that gateway's models; DYNAMO_HEAD's is the
main gateway, which forwards other nodes' models to their gateway), OUT_DIR/nodes.json, and
$STATE_DIR/prometheus-sd/dynamo.json. Everything runs on the host network; etcd, the frontends and the engines' system
APIs listen on 127.0.0.1 (Dynamo has no authentication of its own), and only the gateways (with the tokens) and the
vLLM rendezvous listen on the node's address. Env: STATE_DIR, SERVICE_UID, SERVICE_GID, DYNAMO_HEAD, DYNAMO_WORKER,
MLFLOW_TRACKING_URI/USERNAME/PASSWORD (+ MLFLOW_TRACKING_SERVER_CERT_PATH), and what models.yaml names with $VAR.
"""
import base64
import hashlib
import json
import os
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request

import yaml

here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
spec = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "models.yaml")
cfg = yaml.safe_load(open(spec))
state = os.environ["STATE_DIR"]
HEAD, WORKER = os.environ["DYNAMO_HEAD"], os.environ["DYNAMO_WORKER"]
user = f'{os.environ["SERVICE_UID"]}:{os.environ["SERVICE_GID"]}'
GATEWAY_PORT = 8199   # each node's gateway on its address; the head's is the one Caddy's llm.<host> reaches; any free port


def addr(node):
    return socket.gethostbyname(node)


def mlflow(path, **query):
    url = os.environ["MLFLOW_TRACKING_URI"].rstrip("/") + "/api/2.0/mlflow/" + path + "?" + urllib.parse.urlencode(query)
    auth = base64.b64encode(f'{os.environ["MLFLOW_TRACKING_USERNAME"]}:{os.environ["MLFLOW_TRACKING_PASSWORD"]}'.encode())
    ctx = ssl.create_default_context(cafile=os.environ.get("MLFLOW_TRACKING_SERVER_CERT_PATH"))
    req = urllib.request.Request(url, headers={"Authorization": "Basic " + auth.decode()})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as r:
        return json.load(r)


# What a spec may say. Images only from Dynamo's own registry; per model only these keys; no env var that changes what
# the engine process loads (LD_*, PYTHON*). Mounts, devices and privileges are render.py's alone, never the spec's.
IMAGES = ("nvcr.io/nvidia/ai-dynamo/",)
TOP_KEYS = {"host_numa_nodes", "common_args", "common_env", "gpus", "models"}
MODEL_KEYS = {"name", "registered", "version", "image", "ports", "frontend_args", "args", "env", "licence", "about",
              "placement", "host_numa_nodes"}


def check_spec(cfg):
    bad = set(cfg) - TOP_KEYS
    errs = [f"unknown top-level keys {sorted(bad)}"] if bad else []
    envs = [cfg.get("common_env", {})]
    for m in cfg.get("models", []):
        n = m.get("name", "?")
        if set(m) - MODEL_KEYS:
            errs.append(f"{n}: unknown keys {sorted(set(m) - MODEL_KEYS)}")
        if not str(m.get("image", "")).startswith(IMAGES):
            errs.append(f"{n}: image {m.get('image')!r} is not from {', '.join(IMAGES)}")
        if "version" in m and not str(m["version"]).isdigit():
            errs.append(f"{n}: version {m['version']!r} is not a model version number")
        envs.append(m.get("env", {}))
    errs += [f"env {k} is not allowed" for e in envs for k in e if k.startswith(("LD_", "PYTHON"))]
    if errs:
        raise SystemExit(f"{spec}: " + "; ".join(errs))


def served_version(registered, version):
    """The path, inside the engine container, of the registered model's version's weights."""
    try:
        mv = mlflow("model-versions/get", name=registered, version=str(version))["model_version"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{registered} v{version}: not in the MLflow registry (HTTP {e.code}): register it first")
    src = mv["source"]
    if not src.startswith("mlflow-artifacts:/"):
        raise SystemExit(f"{registered} v{mv['version']}: source {src} is not in the mlflow bucket")
    return "/mlflow/" + src.removeprefix("mlflow-artifacts:/").lstrip("/")


check_spec(cfg)


common = dict(network_mode="host", user=user, ipc="host", shm_size="32gb",
              logging={"driver": "json-file", "options": {"max-size": "20m", "max-file": "5"}})   # 100 MB of logs; untuned


def namespace(m):
    """Each model's own Dynamo namespace: in a shared one, every engine registers at the same endpoint and the second fails."""
    return "flywheel-" + m["name"].replace(".", "-")


def engine(m, path, rank=None):
    """A model's vLLM engine: on the trays, rank 0 or 1 of two; otherwise on its node's GPUs."""
    p = m["ports"]
    if rank is None:
        pl = m["placement"]
        gpus = [str(g) for g in pl["gpus"]]
        place = ["--tensor-parallel-size", str(len(gpus)), "--gpu-memory-utilization", str(pl["share"])]
        devices = [f"nvidia.com/gpu={g}" for g in gpus]
    else:
        place = ["--nnodes", "2", "--node-rank", str(rank), "--master-addr", addr(HEAD), "--master-port", str(p["rendezvous"])]
        place += ["--headless"] if rank else []   # rank 1 only computes; no discovery, no system API
        devices = ["nvidia.com/gpu=all", "/dev/infiniband"]   # NCCL over IB between the trays
    numa = m.get("host_numa_nodes", cfg["host_numa_nodes"])   # host memory on the CPU nodes only (models.yaml); "" off
    cmd = [*(["numactl", "--membind", numa] if numa else []),
           "python3", "-m", "dynamo.vllm", "--model", path, "--served-model-name", m["name"],
           *cfg["common_args"], *m["args"], *place]
    env = {**cfg["common_env"], **m.get("env", {}), "HOME": "/cache", "HF_HOME": "/cache/hf",
           "ETCD_ENDPOINTS": "http://127.0.0.1:2379", "DYN_REQUEST_PLANE": "tcp", "DYN_EVENT_PLANE": "zmq",
           "DYN_TCP_RPC_HOST": "127.0.0.1", "DYN_SYSTEM_HOST": "127.0.0.1", "DYN_SYSTEM_PORT": str(p["system"]),
           "DYN_SYSTEM_ENABLED": "true", "DYN_NAMESPACE": namespace(m)}
    health = {"test": ["CMD", "python3", "-c", f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{p['system']}/health')"],
              "interval": "30s", "timeout": "10s", "retries": 3, "start_period": "60m"}   # start: up to ~25 min with compiling
    return {**common, "image": m["image"], "restart": "no", "command": cmd, "environment": env,
            **({"healthcheck": health} if rank is None else {}),   # engines sharing a GPU wait on it (below)
            "devices": devices, "cap_add": ["IPC_LOCK", "SYS_NICE"],   # IB; numactl
            "ulimits": {"memlock": -1, "stack": 67108864},
            "volumes": [f"{state}/versitygw/buckets/mlflow:/mlflow:ro", f"{state}/dynamo/cache/{m['name']}:/cache"]}


def info(m):
    """What users see per model (landing page, /v1/models): licence, about, and its inputs, read from its flags."""
    flags = [*m.get("args", []), *m.get("frontend_args", [])]
    mm = flags[flags.index("--limit-mm-per-prompt") + 1] if "--limit-mm-per-prompt" in flags else "{}"
    return {"licence": m.get("licence", ""), "about": m.get("about", ""), "images": "--enable-multimodal" in flags,
            "max_images": json.loads(mm).get("image"), "reasoning": "--reasoning-parser" in flags,
            "tools": "--enable-auto-tool-choice" in flags or "--dyn-tool-call-parser" in flags}


def frontend(m):
    p = m["ports"]
    return {**common, "image": m["image"], "restart": "unless-stopped",
            "command": ["python3", "-m", "dynamo.frontend", "--http-host", "127.0.0.1", "--http-port", str(p["frontend"]),
                        "--router-mode", "round-robin", "--namespace", namespace(m), *m.get("frontend_args", [])],
            "environment": {"ETCD_ENDPOINTS": "http://127.0.0.1:2379", "DYN_REQUEST_PLANE": "tcp", "DYN_EVENT_PLANE": "zmq",
                            "HOME": "/tmp"},
            "depends_on": ["etcd"],
            # Dynamo's own chat processor reads the tokenizer and config from the engine's model path (vLLM's gets them
            # from the engine), so only its frontends see the weights' directory
            **({"volumes": [f"{state}/versitygw/buckets/mlflow:/mlflow:ro"]}
               if _flag(m.get("frontend_args", []), "--dyn-chat-processor") == "dynamo" else {})}


def _flag(flags, name):
    return flags[flags.index(name) + 1] if name in flags else None


def etcd(node):
    return {**common, "image": "bitnamilegacy/etcd:3.6.1", "restart": "unless-stopped",
            "environment": {"ALLOW_NONE_AUTHENTICATION": "yes", "ETCD_LISTEN_CLIENT_URLS": "http://127.0.0.1:2379",
                            "ETCD_ADVERTISE_CLIENT_URLS": "http://127.0.0.1:2379",
                            "ETCD_LISTEN_PEER_URLS": "http://127.0.0.1:2380"},
            "volumes": [f"{state}/dynamo/etcd-{node}:/bitnami/etcd"]}   # the image's data path; ETCD_DATA_DIR isn't honoured


def gateway(node):
    return {**common, "build": os.path.join(here, "..", "llm-gateway"), "image": "flywheel-llm-gateway:latest",
            "restart": "unless-stopped",
            "environment": {"GATEWAY_CONFIG": f"/config/{node}.gateway.json", "GATEWAY_LISTEN": f"{addr(node)}:{GATEWAY_PORT}",
                            "LLM_CRITICAL_TOKEN": "${LLM_CRITICAL_TOKEN}"},   # the only token: for X-Priority: critical
            "volumes": [f"{out}:/config:ro"]}


services, local, served, skipped, trays, share, on_gpu, infos = {}, {}, {}, [], [], {}, {}, {}
for m in cfg["models"]:
    if "version" not in m:
        skipped.append(m["name"])
        continue
    path = served_version(m["registered"], m["version"])
    n, p, pl = m["name"], m["ports"], m.get("placement", "trays")
    if pl == "trays":
        trays.append(n)
        node = HEAD
        services.setdefault(WORKER, {})[f"engine-{n}"] = engine(m, path, 1)
        services.setdefault(HEAD, {})[f"engine-{n}"] = {**engine(m, path, 0), "depends_on": ["etcd"]}
    else:
        node = pl["node"] = os.path.expandvars(pl["node"])
        # engines that share a GPU start one after another: vLLM sizes its KV cache from the GPU's free memory, and a
        # neighbour allocating meanwhile counts as its own use (two 27B models started together both failed)
        after = {f"engine-{o}": {"condition": "service_healthy"} for g in pl["gpus"] for o in on_gpu.get((node, str(g)), [])}
        for g in pl["gpus"]:
            share[(node, str(g))] = share.get((node, str(g)), 0) + pl["share"]
            on_gpu.setdefault((node, str(g)), []).append(n)
        services.setdefault(node, {})[f"engine-{n}"] = {**engine(m, path),
                                                        "depends_on": {"etcd": {"condition": "service_started"}, **after}}
    services[node][f"frontend-{n}"] = frontend(m)
    local.setdefault(node, {})[n] = {"frontend": f"http://127.0.0.1:{p['frontend']}", "system": f"http://127.0.0.1:{p['system']}",
                                     "info": info(m)}
    infos[n] = info(m)
    served[n] = {"node": node, "system": p["system"], "tray": pl == "trays"}
if len(trays) > 1:
    raise SystemExit(f"{', '.join(trays)} all have a version: give one of the tray models a version (each takes all 8 GPUs)")
on_trays = [n for n, sv in served.items() if not sv["tray"] and sv["node"] in (HEAD, WORKER)]
if trays and on_trays:
    raise SystemExit(f"{trays[0]}, which takes both trays, and {', '.join(on_trays)}, placed on a tray, all have a version: "
                     "serve one or the other")
budget = {(os.path.expandvars(nd), str(g)): b for nd, gs in cfg.get("gpus", {}).items() for g, b in gs.items()}
for (node, g), s in share.items():
    if s > budget.get((node, g), 0) + 1e-9:
        raise SystemExit(f"{node} GPU {g}: the served models' shares add up to {s:.2f}, over its budget "
                         f"{budget.get((node, g), 0)} (the spec's gpus)")

services.setdefault(HEAD, {})   # the main gateway runs on the head even when only other nodes serve models
for node, svc in services.items():
    if node in local or node == HEAD:   # discovery, frontends and the node's gateway
        svc["etcd"] = etcd(node)
        svc["gateway"] = gateway(node)
os.makedirs(out, exist_ok=True)
for f in os.listdir(out):   # the files of nodes that no longer run anything
    if f.endswith(".json"):
        os.remove(os.path.join(out, f))
known = [m["name"] for m in cfg["models"]]
for node in services:
    if node in local or node == HEAD:
        gw = {"models": dict(local.get(node, {})), "known": known}
        if node == HEAD:   # the main gateway: other nodes' models through their gateways
            gw["models"].update({n: {"gateway": f"http://{addr(sv['node'])}:{GATEWAY_PORT}", "info": infos[n]}
                                 for n, sv in served.items() if sv["node"] != HEAD})
        json.dump(gw, open(os.path.join(out, f"{node}.gateway.json"), "w"), indent=1)
        # the gateway reads its models at start: a changed list changes its environment, so compose restarts it
        services[node]["gateway"]["environment"]["GATEWAY_CONFIG_SHA"] = hashlib.sha256(json.dumps(gw, sort_keys=True).encode()).hexdigest()[:12]
for node, svc in services.items():
    json.dump({"name": "flywheel-dynamo", "services": svc}, open(os.path.join(out, f"{node}.json"), "w"), indent=1)
json.dump({"nodes": sorted(services), "served": served, "trays": trays}, open(os.path.join(out, "nodes.json"), "w"), indent=1)
# Prometheus job dynamo: each model's frontend and engine metrics, relayed by its node's gateway (127.0.0.1 is out of reach).
sd = [{"targets": [f"{addr(sv['node'])}:{GATEWAY_PORT}"],
       "labels": {"model": n, "part": part, "__metrics_path__": f"/metrics/{n}/{part}"}}
      for n, sv in served.items() for part in ("frontend", "engine")]
sd += [{"targets": [f"{addr(node)}:{GATEWAY_PORT}"], "labels": {"part": "gateway"}}   # the other nodes' gateways (served models,
       for node in sorted(local) if node != HEAD]                                    # requests); the head's is job llm-gateway
# Prometheus's live targets only when rendering the live config (ctl.sh dynamo render/up); a test render elsewhere keeps
# its copy beside its other files, so it can't change what Prometheus scrapes.
live = os.path.realpath(out) == os.path.realpath(os.path.join(state, "dynamo"))
sd_dir = os.path.join(state, "prometheus-sd") if live else out
os.makedirs(sd_dir, exist_ok=True)
json.dump(sd, open(os.path.join(sd_dir, "dynamo.json" if live else "prometheus-sd-dynamo.json"), "w"), indent=1)
print(f"rendered {', '.join(f"{n} ({sv['node']})" for n, sv in served.items()) or 'no models'}"
      + (f"; no version (not served): {', '.join(skipped)}" if skipped else ""))
