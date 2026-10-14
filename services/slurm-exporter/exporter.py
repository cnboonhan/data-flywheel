"""Which GPU each Slurm job holds, as Prometheus metrics, so dcgm-exporter's per-GPU series can be joined to jobs.

    slurm_job_gpu{instance="<node>", gpu="<index>", job_id, user, job_name, partition} 1

One series per allocated GPU of every running job this account can see (the cluster hides other users' jobs, so the
service account's own). `instance` and `gpu` match dcgm-exporter's labels, hence honor_labels in prometheus.yml.
"""
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BIN = os.environ.get("SLURM_BIN", "/cm/local/apps/slurm/current/bin")
PORT = 9105   # a free port in the exporter range; compose network only
CACHE_S = 10  # at most one scontrol call per this many seconds, however often it is scraped (Prometheus: 15 s)
_cache = (0.0, "")


def run(*args):
    return subprocess.run([f"{BIN}/{args[0]}", *args[1:]], capture_output=True, text=True, timeout=30).stdout


def indices(idx):
    """'0-1,3' -> ['0', '1', '3']"""
    out = []
    for part in filter(None, idx.split(",")):
        a, _, b = part.partition("-")
        out += [str(i) for i in range(int(a), int(b or a) + 1)]
    return out


def esc(v):
    return v.replace("\\", "\\\\").replace('"', '\\"')


def metrics():
    lines = ["# HELP slurm_job_gpu 1 for each GPU a running Slurm job holds.", "# TYPE slurm_job_gpu gauge"]
    ok = 1
    try:
        text = run("scontrol", "show", "job", "-d", "--oneliner")
    except Exception:   # noqa: BLE001  scontrol missing or hung: report it, keep serving
        text, ok = "", 0
    for job in text.splitlines():
        f = dict(re.findall(r"(\w+)=(\S*)", job))
        if f.get("JobState") != "RUNNING":
            continue
        user = f.get("UserId", "").split("(")[0]
        # the detail part: "Nodes=<hostlist> CPU_IDs=... GRES=gpu:b300:2(IDX:0-1)", once per group of nodes
        for hosts, gres in re.findall(r"Nodes=(\S+) CPU_IDs=\S+ Mem=\S+ GRES=(\S*)", job):
            idx = re.search(r"\(IDX:([^)]*)\)", gres)
            if not idx:
                continue
            for node in run("scontrol", "show", "hostnames", hosts).split():
                for gpu in indices(idx.group(1)):
                    labels = dict(instance=node, gpu=gpu, job_id=f.get("JobId", ""), user=user,
                                  job_name=f.get("JobName", ""), partition=f.get("Partition", ""))
                    lines.append("slurm_job_gpu{%s} 1" % ",".join(f'{k}="{esc(v)}"' for k, v in labels.items()))
    lines += ["# HELP slurm_job_gpu_up 1 if scontrol answered.", "# TYPE slurm_job_gpu_up gauge", f"slurm_job_gpu_up {ok}"]
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _cache
        if self.path != "/metrics":
            self.send_response(404); self.end_headers(); return
        if time.time() - _cache[0] > CACHE_S:
            _cache = (time.time(), metrics())
        body = _cache[1].encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
