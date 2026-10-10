#!/usr/bin/env python3
"""Serve MLflow registered model versions on Triton. Run by the Gitea workflow sync-triton.

Desired state is the MLflow registry: every registered model whose alias `triton` points at a version is served,
as a Triton model with the registered model's name and the MLflow version number as Triton version. Each sync

  1. packages new or changed versions into s3://triton/models/<name>/ (Triton's model repository), using the packager
     for the run's `model` param (PACKAGERS below), and removes models whose alias is gone;
  2. loads or unloads them in Triton (explicit model control) and runs one zero-input inference on each;
  3. records the outcome where people look:
       - MLflow model version tags   triton.status (ready | failed | unsupported | unloaded), triton.model,
                                     triton.synced_at, triton.error
       - Triton model config         parameters mlflow_model, mlflow_version, mlflow_run_id, model, embodiment, action,
                                     mix, data_fingerprint, git_commit, synced_at (GET /v2/models/<name>/config)
       - s3://triton/sync/status.json   the current table; sync/log/<time>.json   every sync that changed something

    python sync.py [--dry-run] [--force NAME ...]
    python sync.py --status          print what Triton has loaded and where each model came from

Env: MLFLOW_TRACKING_URI/USERNAME/PASSWORD, S3_ENDPOINT_URL + AWS_* (the gateway), TRITON_URL (http://triton:8000),
FLYWHEEL_ROOT (the checkout: XPolicyLab code for ACT, models/mlp for MLP).
"""

import argparse
import datetime
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

import boto3
import mlflow

ALIAS = os.environ.get("TRITON_ALIAS", "triton")
BUCKET = os.environ.get("TRITON_BUCKET", "triton")
MODELS, SYNC = "models", "sync"   # the model repository Triton reads, and the sync's own records
TRITON = os.environ.get("TRITON_URL", "http://triton:8000").rstrip("/")
ROOT = os.environ.get("FLYWHEEL_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
XPL = os.path.join(ROOT, "eval", "system1", "RoboDojo", "XPolicyLab")
HERE = os.path.dirname(os.path.abspath(__file__))
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- packagers: (MLflow version, run, downloaded checkpoints dir, target model dir) -> Triton model ----------------

def pbtxt(backend, inputs, outputs, params, max_batch=8, platform=None):
    """config.pbtxt; inputs/outputs: (name, TYPE, dims) without the batch dim."""
    t = lambda kind, n, ty, d: f'{kind} {{ name: "{n}" data_type: TYPE_{ty} dims: [ {", ".join(map(str, d))} ] }}'
    lines = [f'platform: "{platform}"' if platform else f'backend: "{backend}"', f"max_batch_size: {max_batch}"]
    lines += [t("input", *i) for i in inputs] + [t("output", *o) for o in outputs]
    lines += ["dynamic_batching { }", 'instance_group [ { kind: KIND_GPU } ]']   # one instance on every GPU
    lines += [f'parameters {{ key: "{k}" value {{ string_value: {json.dumps(str(v))} }} }}' for k, v in params.items()]
    return "\n".join(lines) + "\n"


def train_args(run):
    """The training command's --key value pairs (later ones win, as with argparse)."""
    words, out = shlex.split(run.data.params.get("command", "")), {}
    for i, w in enumerate(words):
        if w.startswith("--") and i + 1 < len(words) and not words[i + 1].startswith("--"):
            out[w[2:]] = words[i + 1]
    return out


def package_act(mv, run, ckpt, dst, params):
    import yaml
    act = os.path.join(XPL, "policy", "ACT")
    cfg = yaml.safe_load(open(os.path.join(act, "deploy.yml")))
    args = train_args(run)
    num = lambda v: float(v) if "." in v or "e" in v else int(v)
    for k in ("kl_weight", "chunk_size", "hidden_dim", "dim_feedforward", "lr"):
        if k in args:
            cfg[k] = num(args[k])
    tasks = json.load(open(os.path.join(act, "TASK_CONFIGS.json")))
    cfg["camera_names"] = tasks.get(args.get("ckpt_setting", ""), {}).get("camera_names", cfg["camera_names"])
    cfg["action_dim"] = int(run.data.params["action_dim"])
    for f in ("policy_last.ckpt", "dataset_stats.pkl"):
        if not os.path.exists(os.path.join(ckpt, f)):
            raise ValueError(f"checkpoints/ has no {f}")
    v = os.path.join(dst, mv.version)
    os.makedirs(v)
    for f in ("policy_last.ckpt", "dataset_stats.pkl"):
        shutil.copy(os.path.join(ckpt, f), v)
    shutil.copytree(os.path.join(act, "detr"), os.path.join(v, "detr"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info", "setup.py", "LICENSE"))
    shutil.copy(os.path.join(HERE, "backends", "xpolicylab_act.py"), os.path.join(v, "model.py"))
    json.dump(cfg, open(os.path.join(v, "act.json"), "w"), indent=1)
    d = cfg["action_dim"]
    inputs = [("state", "FP32", [d])] + [(c, "UINT8", [-1, -1, 3]) for c in cfg["camera_names"]]
    return pbtxt("python", inputs, [("action", "FP32", [cfg["chunk_size"], d])], params)


def package_mlp(mv, run, ckpt, dst, params):
    import torch
    sys.path[:0] = [os.path.join(ROOT, "services", "slurm", "models", "mlp"), os.path.join(ROOT, "services", "slurm", "lib")]
    from train import MLP
    c = torch.load(os.path.join(ckpt, "final.pt"), map_location="cpu", weights_only=False)
    sd = c["model"]
    d_in, d_out = sd["net.0.weight"].shape[1], sd["net.4.weight"].shape[0]
    mlp = MLP(d_in, d_out, c["args"]["hidden"])
    mlp.load_state_dict(sd)

    class Served(torch.nn.Module):   # normalisation inside the graph; [B, D] -> [B, 1, D] (a chunk of one)
        def __init__(self):
            super().__init__()
            self.mlp = mlp
            self.register_buffer("mean", torch.as_tensor(c["state_mean"]))
            self.register_buffer("std", torch.as_tensor(c["state_std"]))

        def forward(self, state):
            return self.mlp((state - self.mean) / self.std).unsqueeze(1)

    v = os.path.join(dst, mv.version)
    os.makedirs(v)
    torch.onnx.export(Served().eval(), (torch.zeros(1, d_in),), os.path.join(v, "model.onnx"),
                      input_names=["state"], output_names=["action"], opset_version=18,
                      dynamic_shapes={"state": {0: torch.export.Dim("batch", max=4096)}})
    return pbtxt(None, [("state", "FP32", [d_in])], [("action", "FP32", [1, d_out])], params, max_batch=256,
                 platform="onnxruntime_onnx")


PACKAGERS = {"ACT": package_act, "MLP": package_mlp}   # keyed by the training run's param `model`


# ---- S3 ----------------------------------------------------------------------------------------------------------

s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"))


def ensure_bucket():
    try:
        s3.head_bucket(Bucket=BUCKET)
    except Exception:
        s3.create_bucket(Bucket=BUCKET)
    s3.put_object(Bucket=BUCKET, Key=f"{MODELS}/.keep", Body=b"")   # the gateway deletes a folder once it's empty


def keys(prefix):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        yield from (o["Key"] for o in page.get("Contents", []))


def served():
    """{name: provenance} for the models in the repository (from each model's flywheel.json)."""
    out = {}
    for k in keys(f"{MODELS}/"):
        parts = k.split("/")
        if len(parts) == 3 and parts[2] == "flywheel.json":
            out[parts[1]] = json.loads(s3.get_object(Bucket=BUCKET, Key=k)["Body"].read())
    return out


def remove(name):
    for k in list(keys(f"{MODELS}/{name}/")):
        s3.delete_object(Bucket=BUCKET, Key=k)


def upload(src, name):
    remove(name)
    for dirpath, _, files in os.walk(src):
        for f in files:
            p = os.path.join(dirpath, f)
            s3.upload_file(p, BUCKET, f"{MODELS}/{name}/{os.path.relpath(p, src)}")


def put_json(key, obj):
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(obj, indent=1).encode(), ContentType="application/json")


# ---- Triton ------------------------------------------------------------------------------------------------------

def triton(path, body=None, method=None):
    req = urllib.request.Request(TRITON + path, data=None if body is None else json.dumps(body).encode(),
                                 method=method or ("POST" if body is not None else "GET"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw.decode(errors="replace")}


def smoke_test(name):
    """One inference with zero inputs (images 64x64); returns an error string or None."""
    code, cfg = triton(f"/v2/models/{name}/config")
    if code != 200:
        return cfg.get("error", f"HTTP {code}")
    inputs = []
    for i in cfg["input"]:
        dims = [64 if d == -1 else int(d) for d in i["dims"]]
        n = 1
        for d in dims:
            n *= d
        dtype = i["data_type"].removeprefix("TYPE_")
        inputs.append({"name": i["name"], "shape": [1, *dims], "datatype": dtype, "data": [0] * n})
    code, out = triton(f"/v2/models/{name}/infer", {"inputs": inputs})
    if code != 200:
        return out.get("error", f"HTTP {code}")
    shape = out["outputs"][0]["shape"]
    print(f"  {name}: inference ok, {out['outputs'][0]['name']} {shape}")
    return None


# ---- sync --------------------------------------------------------------------------------------------------------

def desired(client):
    """{name: ModelVersion} for every registered model with the alias."""
    out = {}
    for rm in client.search_registered_models(max_results=1000):
        if ALIAS in (rm.aliases or {}):
            out[rm.name] = client.get_model_version(rm.name, rm.aliases[ALIAS])
    return out


def tag(client, name, version, status, error=""):
    for k, v in {"triton.status": status, "triton.model": name, "triton.synced_at": NOW, "triton.error": error[:5000]}.items():
        try:
            client.set_model_version_tag(name, str(version), k, v)
        except Exception as e:   # the version may have been deleted
            print(f"  warning: could not tag {name} v{version}: {e}")


def sync(args):
    client = mlflow.MlflowClient()
    ensure_bucket()
    want, have = desired(client), served()
    changes, rows, status = [], [], []   # rows: (name, version, action) to bring up in Triton; status: what we report

    for name, prov in have.items():
        if name not in want:
            print(f"- {name} v{prov['mlflow_version']}: alias '{ALIAS}' removed, unloading")
            if not args.dry_run:
                triton(f"/v2/repository/models/{name}/unload", {})
                remove(name)
                tag(client, name, prov["mlflow_version"], "unloaded")
            changes.append({"model": name, "version": prov["mlflow_version"], "action": "unload"})

    for name, mv in sorted(want.items()):
        prov = have.get(name)
        current = prov and prov["mlflow_version"] == mv.version and prov["mlflow_run_id"] == mv.run_id
        if current and name not in args.force:
            rows.append((name, mv.version, None))
            continue
        if not current and mv.tags.get("triton.error", "").startswith("packaging") and name not in args.force:
            status.append({"model": name, "version": mv.version, "status": "failed", "error": mv.tags["triton.error"]})
            continue   # packaging failed before; retried only with --force (or a new version)
        run = client.get_run(mv.run_id) if mv.run_id else None
        kind = run.data.params.get("model") if run else None
        packager = PACKAGERS.get(kind)
        if packager is None:
            msg = f"no packager for model '{kind}' (have: {', '.join(PACKAGERS)})"
            print(f"! {name} v{mv.version}: {msg}")
            if not args.dry_run and mv.tags.get("triton.status") != "unsupported":   # tag once, not every sync
                tag(client, name, mv.version, "unsupported", msg)
            status.append({"model": name, "version": mv.version, "status": "unsupported", "error": msg})
            continue
        print(f"+ {name} v{mv.version} ({kind}, run {mv.run_id}){' replaces v' + prov['mlflow_version'] if prov else ''}")
        changes.append({"model": name, "version": mv.version, "action": "load", "replaces": prov and prov["mlflow_version"]})
        if args.dry_run:
            continue
        p = run.data.params
        params = {"mlflow_model": name, "mlflow_version": mv.version, "mlflow_run_id": mv.run_id, "model": kind,
                  "embodiment": p.get("embodiment", ""), "action": p.get("action", ""), "mix": p.get("mix", ""),
                  "data_fingerprint": p.get("data_fingerprint", ""), "git_commit": p.get("git_commit", ""), "synced_at": NOW}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ckpt = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=os.path.join(tmp, "ckpt"))
                dst = os.path.join(tmp, "repo")
                os.makedirs(dst)
                cfg = packager(mv, run, ckpt, dst, params)
                open(os.path.join(dst, "config.pbtxt"), "w").write(cfg)
                json.dump(params, open(os.path.join(dst, "flywheel.json"), "w"), indent=1)
                upload(dst, name)
            rows.append((name, mv.version, "load"))
        except Exception as e:
            print(f"  packaging failed: {type(e).__name__}: {e}")
            err = f"packaging: {type(e).__name__}: {e}"
            tag(client, name, mv.version, "failed", err)
            status.append({"model": name, "version": mv.version, "status": "failed", "error": err})
            for c in changes:
                if c["model"] == name:
                    c["status"], c["error"] = "failed", err

    if args.dry_run:
        return
    for name, version, action in rows:
        err = None
        if action == "load":
            code, out = triton(f"/v2/repository/models/{name}/load", {})
            err = None if code == 200 else out.get("error", f"HTTP {code}")
        else:
            code, _ = triton(f"/v2/models/{name}/ready")
            if code != 200:   # in the repository but not loaded (e.g. Triton restarted and failed it): try again
                code, out = triton(f"/v2/repository/models/{name}/load", {})
                err = None if code == 200 else out.get("error", f"HTTP {code}")
                action = "reload"
        err = err or smoke_test(name)
        state = "failed" if err else "ready"
        if action or err:
            print(f"  {name} v{version}: {state}{': ' + err if err else ''}")
            tag(client, name, version, state, err or "")
        status.append({"model": name, "version": version, "status": state, "error": err or ""})
        if action == "reload":
            changes.append({"model": name, "version": version, "action": "reload"})
        for c in changes:
            if c["model"] == name and c["action"] in ("load", "reload"):
                c["status"], c["error"] = state, err or ""

    put_json(f"{SYNC}/status.json", {"synced_at": NOW, "alias": ALIAS, "models": status})
    if changes:
        put_json(f"{SYNC}/log/{NOW}.json", {"synced_at": NOW, "changes": changes})
    print(f"{len(status)} model(s) served, {len(changes)} change(s)")
    if any(s["status"] == "failed" for s in status):
        sys.exit(1)


def show_status():
    code, index = triton("/v2/repository/index", {})
    if code != 200:
        sys.exit(f"Triton: HTTP {code} {index}")
    print(f"{'model':48} {'ver':>4} {'state':12} {'mlflow version':>14}  run / embodiment / mix")
    for m in index:
        code, cfg = triton(f"/v2/models/{m['name']}/config") if m.get("state") == "READY" else (0, {})
        p = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        print(f"{m['name']:48} {m.get('version', ''):>4} {m.get('state', ''):12} {p.get('mlflow_version', '-'):>14}  "
              f"{p.get('mlflow_run_id', '-')} / {p.get('embodiment', '-')} / {p.get('mix', '-')}"
              + (f"  ({m['reason']})" if m.get("reason") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report what would change, touch nothing")
    ap.add_argument("--force", nargs="*", default=[], help="repackage and reload these models even if unchanged")
    ap.add_argument("--status", action="store_true", help="print Triton's loaded models with their MLflow source")
    args = ap.parse_args()
    show_status() if args.status else sync(args)


if __name__ == "__main__":
    main()
