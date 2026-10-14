"""Push what each running Slurm job holds and uses to a Prometheus Pushgateway, so node metrics can be joined to jobs.

    slurm_job_gpu{instance="<node>", gpu="<index>", <job>} 1      each GPU it holds (joins dcgm-exporter on instance, gpu)
    slurm_job_cpu{instance="<node>", cpu="<index>", <job>} 1      each CPU core it is pinned to (joins node_exporter's
                                                                  node_cpu_seconds_total on instance, cpu)
    slurm_job_cpus{<job>}, slurm_job_memory_bytes{<job>}         what it was allocated
    slurm_job_rss_bytes{<job>}                                   memory in use, summed over its running steps (sstat)
    <job> = job_id, user, job_name, partition

    PUSHGATEWAY_TOKEN=<token> python3 exporter.py --push https://<host>:<port>/pushgateway [SEC]

Every running job the running account can see (an account that sees all jobs publishes everyone's). It rewrites
the group job="slurm_jobs" (PUSHGATEWAY_JOB) every SEC seconds (30) and deletes it on exit, so a
stopped pusher leaves no stale jobs behind (a killed one leaves its last push; push_time_seconds dates it). `instance`
and `gpu` match dcgm-exporter's labels; `cpu` is Slurm's CPU_IDs, the OS core numbers wherever Slurm's
numbering matches the OS's (check one job's Cpus_allowed_list). Needs only Python 3, `scontrol` and `sstat` on PATH
(or SLURM_BIN). The token is
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
GROUP = "/metrics/job/" + os.environ.get("PUSHGATEWAY_JOB", "slurm_jobs")   # the group this script owns; a push replaces it whole


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


UNITS = {"K": 2 ** 10, "M": 2 ** 20, "G": 2 ** 30, "T": 2 ** 40}


def size(v):
    """Slurm's memory sizes, '2052M' or '1.50G' (a bare number is MiB) -> bytes"""
    m = re.fullmatch(r"([\d.]+)([KMGT]?)", v or "")
    return float(m.group(1)) * UNITS.get(m.group(2) or "M", 1) if m else 0.0


def rss(job_ids):
    """Memory in use per job, summed over its running steps, from sstat's TRESUsageInTot (one call for all jobs)."""
    out = {}
    if not job_ids:
        return out
    try:
        text = run("sstat", "-a", "--noheader", "-P", "--format=JobID,TRESUsageInTot", "-j", ",".join(job_ids))
    except Exception:   # noqa: BLE001  no sstat: no rss
        return out
    for line in text.splitlines():
        step, _, tres = line.partition("|")
        m = re.search(r"(?:^|,)mem=([^,]+)", tres)
        if m:
            job = step.split(".")[0]
            out[job] = out.get(job, 0.0) + size(m.group(1))
    return out


def metrics():
    gpus, cpus, alloc = [], [], []
    ok = 1
    try:
        text = run("scontrol", "show", "job", "-d", "--oneliner")
    except Exception:   # noqa: BLE001  scontrol missing or hung: report it, keep pushing
        text, ok = "", 0
    for job in text.splitlines():
        f = dict(re.findall(r"(\w+)=(\S*)", job))
        if f.get("JobState") != "RUNNING":
            continue
        jl = ",".join(f'{k}="{esc(v)}"' for k, v in dict(job_id=f.get("JobId", ""), user=f.get("UserId", "").split("(")[0],
                      job_name=f.get("JobName", ""), partition=f.get("Partition", "")).items())
        nodes = int(f.get("NumNodes", "1").split("-")[0] or 1)
        ncpu = int(f.get("NumCPUs", "0") or 0)
        mem = size(f["MinMemoryNode"]) * nodes if "MinMemoryNode" in f else size(f.get("MinMemoryCPU", "")) * ncpu
        alloc.append((f.get("JobId", ""), jl, ncpu, mem))
        # the detail part: "Nodes=<hostlist> CPU_IDs=0-3 Mem=... GRES=gpu:<type>:2(IDX:0-1)", once per group of nodes
        for hosts, cpu_ids, gres in re.findall(r"Nodes=(\S+) CPU_IDs=(\S+) Mem=\S+ GRES=(\S*)", job):
            idx = re.search(r"\(IDX:([^)]*)\)", gres)
            for node in run("scontrol", "show", "hostnames", hosts).split():
                cpus += [f'slurm_job_cpu{{instance="{node}",cpu="{c}",{jl}}} 1' for c in indices(cpu_ids)]
                if idx:
                    gpus += [f'slurm_job_gpu{{instance="{node}",gpu="{g}",{jl}}} 1' for g in indices(idx.group(1))]
    used = rss([j for j, *_ in alloc])
    lines = ["# HELP slurm_job_gpu 1 for each GPU a running Slurm job holds.", "# TYPE slurm_job_gpu gauge", *gpus,
             "# HELP slurm_job_cpu 1 for each CPU core a running Slurm job is pinned to.", "# TYPE slurm_job_cpu gauge", *cpus,
             "# HELP slurm_job_cpus CPUs allocated to the job.", "# TYPE slurm_job_cpus gauge",
             *[f"slurm_job_cpus{{{jl}}} {n}" for _, jl, n, _ in alloc],
             "# HELP slurm_job_memory_bytes Memory allocated to the job (0: none requested).", "# TYPE slurm_job_memory_bytes gauge",
             *[f"slurm_job_memory_bytes{{{jl}}} {m:.0f}" for _, jl, _, m in alloc],
             "# HELP slurm_job_rss_bytes Memory in use by the job's running steps (sstat).", "# TYPE slurm_job_rss_bytes gauge",
             *[f"slurm_job_rss_bytes{{{jl}}} {used[j]:.0f}" for j, jl, _, _ in alloc if j in used],
             "# HELP slurm_job_gpu_up 1 if scontrol answered.", "# TYPE slurm_job_gpu_up gauge", f"slurm_job_gpu_up {ok}"]
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
        jobs = re.findall(r'^slurm_job_cpus\{.*job_id="([^"]*)"', body, re.M)
        gpus = re.findall(r'^slurm_job_gpu\{', body, re.M)
        try:
            status = send(url, "PUT", body.encode())
            log(f"pushed {len(jobs)} jobs holding {len(gpus)} GPUs (HTTP {status})"
                + ("" if "slurm_job_gpu_up 1" in body else "; scontrol failed"))
        except Exception as e:   # noqa: BLE001  a failed push is retried on the next round
            log(f"push failed: {e}")
        time.sleep(every)


if __name__ == "__main__":
    if sys.argv[1:2] != ["--push"] or len(sys.argv) < 3:
        sys.exit(__doc__)
    push(sys.argv[2].rstrip("/"), int(sys.argv[3]) if len(sys.argv) > 3 else 30)
