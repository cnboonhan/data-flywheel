#!/usr/bin/env python3
"""Serve a Triton model repository whose weights live in the MLflow registry. Run by the Gitea workflow sync-triton.

Desired state is the Gitea repo triton, a Triton model repository (Triton's own layout: <model>/config.pbtxt and
<model>/<version>/ with the code; services/triton/models/ is its seed) without the weights: each config.pbtxt names
them as the parameters mlflow_model and mlflow_version, a version an admin registered in MLflow. Each sync

  1. copies new or changed models into s3://triton/models/<name>/ (the repository Triton reads), adding to each
     config.pbtxt the parameters weights (the version's artifacts, read in place under /mlflow, which the server
     mounts), mlflow_run_id and synced_at, and for a vLLM model setting model.json's "model" to the weights; a model
     changes when its files or its MLflow version's run do. Models no longer in the repo are removed;
  2. loads or unloads them in Triton (explicit model control) and checks each is ready (and, when all its required
     inputs are numbers, answers one request of zeros);
  3. records the outcome in s3://triton/sync/status.json (the current table) and sync/log/<time>.json (every sync that
     changed something), and in each model's config (GET /v2/models/<name>/config).

    python sync.py [--repo DIR] [--dry-run] [--force NAME ...]
    python sync.py --status          print what Triton has loaded and where each model came from

Env: MLFLOW_TRACKING_URI/USERNAME/PASSWORD, S3_ENDPOINT_URL + AWS_* (the gateway), TRITON_URL (http://triton:8000),
TRITON_KEY (Triton's restricted APIs), TRITON_REPO (the model repository, else --repo).
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

import boto3
import mlflow

BUCKET = os.environ.get("TRITON_BUCKET", "triton")
MODELS, SYNC = "models", "sync"   # the model repository Triton reads, and the sync's own records
TRITON = os.environ.get("TRITON_URL", "http://triton:8000").rstrip("/")
KEY = {"triton-key": os.environ["TRITON_KEY"]} if os.environ.get("TRITON_KEY") else {}   # Triton's restricted APIs
HERE = os.path.dirname(os.path.abspath(__file__))
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")   # a model directory's name, Triton's model name


# ---- the repository --------------------------------------------------------------------------------------------------

def param(cfg, key):
    """A string parameter of a config.pbtxt, or None."""
    m = re.search(r'parameters\s*\{\s*key:\s*"' + re.escape(key) + r'"\s*value\s*\{\s*string_value:\s*"([^"]*)"', cfg)
    return m and m.group(1)


def read_repo(path):
    """{name: (model dir, config text, mlflow model, version)} for every model in the repository; exits on errors."""
    out, errs = {}, []
    for name in sorted(os.listdir(path)):
        d = os.path.join(path, name)
        if not os.path.isfile(os.path.join(d, "config.pbtxt")):
            continue   # README.md, .git, ...
        cfg = open(os.path.join(d, "config.pbtxt")).read()
        model, version = param(cfg, "mlflow_model"), param(cfg, "mlflow_version")
        if not NAME.fullmatch(name):
            errs.append(f"{name}: not a valid model name")
        if not model or not (version or "").isdigit():
            errs.append(f"{name}: config.pbtxt needs the parameters mlflow_model and mlflow_version (a number)")
        if param(cfg, "weights") is not None:
            errs.append(f"{name}: config.pbtxt sets weights, which sync.py fills in")
        if not any(v.isdigit() for v in os.listdir(d)):
            errs.append(f"{name}: no version directory (<model>/1/)")
        out[name] = (d, cfg, model, version)
    if errs:
        sys.exit(f"{path}: " + "; ".join(errs))
    return out


def fingerprint(d, mv):
    """The model directory's files and the MLflow version's run: a change to either reloads the model."""
    h = hashlib.sha256(f"{mv.name} {mv.version} {mv.run_id}".encode())
    for root, dirs, files in sorted(os.walk(d)):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, d).encode() + open(p, "rb").read())
    return h.hexdigest()[:12]


def package(name, d, cfg, mv, dst):
    """The model as Triton reads it: its files, config.pbtxt with the weights' path and provenance added."""
    weights = "/mlflow/" + mv.source.removeprefix("mlflow-artifacts:/").lstrip("/")
    shutil.copytree(d, dst, ignore=shutil.ignore_patterns("__pycache__"))
    add = {"weights": weights, "mlflow_run_id": mv.run_id or "", "synced_at": NOW}
    cfg = cfg.rstrip("\n") + "\n" + "\n".join(
        f'parameters {{ key: "{k}" value {{ string_value: {json.dumps(v)} }} }}' for k, v in add.items()) + "\n"
    open(os.path.join(dst, "config.pbtxt"), "w").write(cfg)
    if re.search(r'^\s*backend:\s*"vllm"', cfg, re.M):   # vLLM's engine arguments: the weights are its "model"
        for v in os.listdir(dst):
            mj = os.path.join(dst, v, "model.json")
            if os.path.isfile(mj):
                engine = json.load(open(mj))
                engine["model"] = weights
                json.dump(engine, open(mj, "w"), indent=1)


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
                                 headers={"Content-Type": "application/json", **KEY})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:   # a load answers once done: ~4 min for a 122B; untuned
            raw = r.read()
            return r.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": raw.decode(errors="replace")}


NUMERIC = {"FP32", "FP16", "BF16", "FP64", "INT8", "INT16", "INT32", "INT64", "UINT8", "UINT16", "UINT32", "UINT64", "BOOL"}


def smoke_test(name):
    """The model is ready; when every required input is a number, it also answers one request of zeros (dims of -1 as
    64; images 64x64), else readiness is the check (encoded images or text can't be zeros). An error string or None."""
    code, out = triton(f"/v2/models/{name}/ready")
    if code != 200:
        return out.get("error", f"not ready (HTTP {code})")
    code, cfg = triton(f"/v2/models/{name}/config")
    if code != 200:
        return cfg.get("error", f"HTTP {code}")
    if cfg.get("backend") == "vllm":
        code, out = triton(f"/v2/models/{name}/generate",
                           {"text_input": "Hello", "sampling_parameters": json.dumps({"max_tokens": 4})})
        return None if code == 200 else out.get("error", f"HTTP {code}")
    required = [i for i in cfg.get("input", []) if not i.get("optional")]
    if any(i["data_type"].removeprefix("TYPE_") not in NUMERIC for i in required):
        print(f"  {name}: ready (its inputs aren't numbers: no test request)")
        return None
    inputs = []
    for i in required:
        dims = [64 if d == -1 else int(d) for d in i["dims"]]
        n = 1
        for d in dims:
            n *= d
        batch = [1] if cfg.get("max_batch_size", 0) else []   # a model without batching takes its inputs as they are
        inputs.append({"name": i["name"], "shape": [*batch, *dims], "datatype": i["data_type"].removeprefix("TYPE_"),
                       "data": [0] * n})
    code, out = triton(f"/v2/models/{name}/infer", {"inputs": inputs})
    if code != 200:
        return out.get("error", f"HTTP {code}")
    print(f"  {name}: inference ok, {out['outputs'][0]['name']} {out['outputs'][0]['shape']}")
    return None


# ---- sync --------------------------------------------------------------------------------------------------------

def sync(args):
    client = mlflow.MlflowClient()
    ensure_bucket()
    repo, have = read_repo(args.repo), served()
    changes, rows, status = [], [], []   # rows: (name, version, action) to bring up in Triton; status: what we report

    for name, prov in have.items():
        if name not in repo:
            print(f"- {name}: no longer in the repository, unloading")
            if not args.dry_run:
                triton(f"/v2/repository/models/{name}/unload", {})
                remove(name)
            changes.append({"model": name, "version": prov.get("mlflow_version"), "action": "unload"})

    for name, (d, cfg, model, version) in sorted(repo.items()):
        try:
            mv = client.get_model_version(model, version)
        except Exception as e:
            err = f"{model} v{version} is not in the MLflow registry ({type(e).__name__}): register it first"
            print(f"! {name}: {err}")
            status.append({"model": name, "version": version, "status": "failed", "error": err})
            continue
        if not mv.source.startswith("mlflow-artifacts:/"):
            err = f"{model} v{version}: source {mv.source} is not in the registry's artifact store"
            status.append({"model": name, "version": version, "status": "failed", "error": err})
            continue
        fp = fingerprint(d, mv)
        if have.get(name, {}).get("fingerprint") == fp and name not in args.force:
            rows.append((name, version, None))
            continue
        print(f"+ {name}: {model} v{version} (run {mv.run_id}, files {fp})")
        changes.append({"model": name, "version": version, "action": "load"})
        if args.dry_run:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            dst = os.path.join(tmp, name)
            package(name, d, cfg, mv, dst)
            json.dump({"mlflow_model": model, "mlflow_version": version, "mlflow_run_id": mv.run_id, "fingerprint": fp,
                       "synced_at": NOW}, open(os.path.join(dst, "flywheel.json"), "w"), indent=1)
            upload(dst, name)
        rows.append((name, version, "load"))

    if args.dry_run:
        return
    for name, version, action in rows:
        err = None
        if action == "load":
            code, cfg = triton(f"/v2/models/{name}/config")
            if code == 200 and cfg.get("backend") == "vllm":
                # Triton loads the new copy before it drops the old one, and a vLLM model's share of its GPU leaves no
                # room for two: unload first (the model is down until the new copy is ready, ~4 min for a 122B)
                triton(f"/v2/repository/models/{name}/unload", {})
                for _ in range(60):   # wait up to 5 min (tries x 5 s) for the old engine to free its GPU memory
                    if triton(f"/v2/models/{name}/ready")[0] != 200:
                        break
                    time.sleep(5)
                time.sleep(15)   # the engine process exits after Triton marks the model unloaded; untuned
            code, out = triton(f"/v2/repository/models/{name}/load", {})
            err = None if code == 200 else out.get("error", f"HTTP {code}")
        elif triton(f"/v2/models/{name}/ready")[0] != 200:   # in the repository but not loaded (e.g. Triton restarted)
            code, out = triton(f"/v2/repository/models/{name}/load", {})
            err = None if code == 200 else out.get("error", f"HTTP {code}")
            action = "reload"
        err = err or smoke_test(name)
        state = "failed" if err else "ready"
        if action or err:
            print(f"  {name} v{version}: {state}{': ' + err if err else ''}")
        status.append({"model": name, "version": version, "status": state, "error": err or ""})
        if action == "reload":
            changes.append({"model": name, "version": version, "action": "reload"})
        for c in changes:
            if c["model"] == name and c["action"] in ("load", "reload"):
                c["status"], c["error"] = state, err or ""

    put_json(f"{SYNC}/status.json", {"synced_at": NOW, "models": status})
    if changes:
        put_json(f"{SYNC}/log/{NOW}.json", {"synced_at": NOW, "changes": changes})
    print(f"{len(status)} model(s) served, {len(changes)} change(s)")
    if any(s["status"] == "failed" for s in status):
        sys.exit(1)


def show_status():
    code, index = triton("/v2/repository/index", {})
    if code != 200:
        sys.exit(f"Triton: HTTP {code} {index}")
    print(f"{'model':48} {'ver':>4} {'state':12} {'mlflow version':>14}  run")
    for m in index:
        code, cfg = triton(f"/v2/models/{m['name']}/config") if m.get("state") == "READY" else (0, {})
        p = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        print(f"{m['name']:48} {m.get('version', ''):>4} {m.get('state', ''):12} {p.get('mlflow_version', '-'):>14}  "
              f"{p.get('mlflow_run_id', '-')}" + (f"  ({m['reason']})" if m.get("reason") else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.environ.get("TRITON_REPO", os.path.join(HERE, "models")),
                    help="the model repository (the Gitea repo triton; default $TRITON_REPO, else the seed)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, touch nothing")
    ap.add_argument("--force", nargs="*", default=[], help="reload these models even if unchanged")
    ap.add_argument("--status", action="store_true", help="print Triton's loaded models with their MLflow source")
    args = ap.parse_args()
    show_status() if args.status else sync(args)


if __name__ == "__main__":
    main()
