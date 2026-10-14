"""Push which GPU each running Slurm job holds to a Prometheus Pushgateway, so per-GPU metrics can be joined to jobs.

    slurm_job_gpu{instance="<node>", gpu="<index>", job_id, user, job_name, partition} 1

    PUSHGATEWAY_TOKEN=<token> python3 exporter.py --push https://<host>:<port>/pushgateway [SEC]

One series per allocated GPU of every running job the running account can see (an account that sees all jobs
publishes everyone's). It rewrites the group job="slurm_jobs" every SEC seconds (30) and deletes it on exit, so a
stopped pusher leaves no stale jobs behind (a killed one leaves its last push; push_time_seconds dates it). `instance`
and `gpu` match dcgm-exporter's labels. Needs only Python 3 and `scontrol` on PATH (or SLURM_BIN). The token is
read from the environment, not the command line, so `ps` doesn't show it; the CA bundle from PUSHGATEWAY_CA (else
AWS_CA_BUNDLE, else the system's).
"""
import os
import re
import shutil
import signal
import ssl
import subprocess
import sys
import time
import urllib.request

BIN = os.environ.get("SLURM_BIN") or os.path.dirname(shutil.which("scontrol") or "/usr/bin/scontrol")   # Slurm's bin/
GROUP = "/metrics/job/slurm_jobs"   # the Pushgateway group this script owns; a push replaces it whole


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
    except Exception:   # noqa: BLE001  scontrol missing or hung: report it, keep pushing
        text, ok = "", 0
    for job in text.splitlines():
        f = dict(re.findall(r"(\w+)=(\S*)", job))
        if f.get("JobState") != "RUNNING":
            continue
        user = f.get("UserId", "").split("(")[0]
        # the detail part: "Nodes=<hostlist> CPU_IDs=... GRES=gpu:<type>:2(IDX:0-1)", once per group of nodes
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


def send(url, method, body=None):
    req = urllib.request.Request(url + GROUP, data=body, method=method,
                                 headers={"Authorization": "Bearer " + os.environ.get("PUSHGATEWAY_TOKEN", ""),
                                          "Content-Type": "text/plain; version=0.0.4"})
    ca = os.environ.get("PUSHGATEWAY_CA") or os.environ.get("AWS_CA_BUNDLE")
    with urllib.request.urlopen(req, timeout=30, context=ssl.create_default_context(cafile=ca)) as r:
        return r.status


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def push(url, every):
    def stop(*_):
        try:
            send(url, "DELETE")
            log("stopped; deleted the pushed jobs")
        finally:
            sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(f"pushing to {url}{GROUP} every {every} s with {BIN}/scontrol")
    while True:
        body = metrics()
        gpus = re.findall(r'^slurm_job_gpu\{.*job_id="([^"]*)"', body, re.M)
        try:
            status = send(url, "PUT", body.encode())
            log(f"pushed {len(set(gpus))} jobs on {len(gpus)} GPUs (HTTP {status})"
                + ("" if "slurm_job_gpu_up 1" in body else "; scontrol failed"))
        except Exception as e:   # noqa: BLE001  a failed push is retried on the next round
            log(f"push failed: {e}")
        time.sleep(every)


if __name__ == "__main__":
    if sys.argv[1:2] != ["--push"] or len(sys.argv) < 3:
        sys.exit(__doc__)
    push(sys.argv[2].rstrip("/"), int(sys.argv[3]) if len(sys.argv) > 3 else 30)
