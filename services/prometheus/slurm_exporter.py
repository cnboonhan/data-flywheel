"""Push what each running Slurm job holds and uses to a Prometheus Pushgateway, so node metrics can be joined to jobs.

    slurm_job_gpu{instance="<node>", gpu="<index>", <job>} 1      each GPU it holds (joins dcgm-exporter on instance, gpu)
    slurm_job_cpu{instance="<node>", cpu="<index>", <job>} 1      each CPU core it is pinned to (joins node_exporter's
                                                                  node_cpu_seconds_total on instance, cpu)
    slurm_job_cpus{<job>}, slurm_job_memory_bytes{<job>}         what it was allocated
    slurm_job_rss_bytes{<job>}                                   memory in use, summed over its running steps (sstat)
    slurm_job_pending_seconds{<job>, reason}                     the queue: each pending job, how long it has waited
    slurm_job_pending_gpus{<job>, reason}, slurm_job_pending_nodes{...}, slurm_job_pending_priority{...}   what it asks
                                                                  for, and its priority (0: held)
    slurm_node_state{instance, state, reason, partition} 1       each node's state (sinfo: idle, mixed, allocated, drained,
                                                                  down, ...; a trailing * means not responding)
    slurm_node_last_job_start_seconds{instance, <job>, job_state}   the latest job started on the node (sacct), and when
    slurm_node_last_job_end_seconds{instance, <job>, job_state}     when it ended (0 while it runs)
    slurm_node_reason_seconds{instance}                          when Slurm set a down or drained node's reason
    slurm_node_cause_job_seconds{instance, cause_job, cause_user, cause_name, cause_state}   the job that tripped it, and
                                                                  when: the one Slurm requeued or failed on it then (a
                                                                  prolog error requeues the job it was starting), else
                                                                  the one running then; none when the node was empty
    <job> = job_id, user, job_name, partition

    PUSHGATEWAY_TOKEN=<token> python3 slurm_exporter.py --push https://<host>:<port>/pushgateway [SEC]

Every running job the running account can see (an account that sees all jobs publishes everyone's). It rewrites
the group job="slurm_jobs" (PUSHGATEWAY_JOB) every SEC seconds (30) and deletes it on exit, so a
stopped pusher leaves no stale jobs behind (a killed one leaves its last push; push_time_seconds dates it). `instance`
and `gpu` match dcgm-exporter's labels; `cpu` is Slurm's CPU_IDs, the OS core numbers wherever Slurm's
numbering matches the OS's (check one job's Cpus_allowed_list). Needs only Python 3, `scontrol`, `sstat`, `sinfo` and
`sacct` on PATH (or SLURM_BIN). The token is
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
RECENT_DAYS = 2      # the last job per node: jobs of the last 2 days for every node at once (~0.1 s of sacct); untuned
LOOKBACK_DAYS = 30   # nodes without one: searched this far back, by node; older counts as none


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


def hosts(nodelist):
    """'C2-GB300-01-C[05-06,09]' -> ['C2-GB300-01-C05', 'C2-GB300-01-C06', 'C2-GB300-01-C09']; scontrol for anything else."""
    m = re.fullmatch(r"([^\[\],]+)\[([\d,-]+)\]", nodelist)
    if not m:
        return run("scontrol", "show", "hostnames", nodelist).split() if "[" in nodelist else nodelist.split(",")
    return [f"{m.group(1)}{i:0{len(a)}d}" for a, _, b in (p.partition("-") for p in m.group(2).split(","))
            for i in range(int(a), int(b or a) + 1)]


def seconds(t):
    """sacct's local time, '2026-10-17T10:26:35', to epoch seconds; 0 for Unknown/None (running, never started)"""
    try:
        return time.mktime(time.strptime(t, "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return 0.0


def last_jobs(nodes, days, only=None):
    """The latest-started job on each node (sacct -X: jobs, not their steps)."""
    args = ["sacct", "-a", "-X", "-n", "-P", "-S", f"now-{days}days", "-E", "now",
            "-o", "JobID,User,JobName,Partition,State,Start,End,NodeList", *(["-N", ",".join(only)] if only else [])]
    last = {}
    for line in run(*args).splitlines():
        f = line.split("|")
        if len(f) != 8 or not seconds(f[5]) or f[7] in ("", "None assigned"):
            continue
        for node in hosts(f[7]):
            if node in nodes and (node not in last or seconds(f[5]) > seconds(last[node][5])):
                last[node] = f
    return last


CAUSE_WINDOW_S = 120   # a requeued or failed record this close to the reason's time counts as its cause; untuned
DOWN = re.compile(r"down|drain|fail|\*$")   # sinfo states for a node out of service (* = not responding)


def stamp(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def cause_job(node, t):
    """The job that tripped the node's fall at t: a record Slurm requeued or failed there then (sacct -D shows the record a
    requeue leaves on the node), else the job running then, else None."""
    out = run("sacct", "-a", "-D", "-X", "-n", "-P", "-N", node, "-S", stamp(t - CAUSE_WINDOW_S), "-E", stamp(t + 60),
              "-o", "JobID,User,JobName,Partition,State,Start,End,NodeList")
    recs = [f for f in (line.split("|") for line in out.splitlines()) if len(f) == 8]
    failed = [f for f in recs if re.match(r"REQUEUED|NODE_FAIL|FAILED|BOOT_FAIL|LAUNCH_FAILED", f[4])
              and abs((seconds(f[6]) or t) - t) <= CAUSE_WINDOW_S]
    if failed:
        return min(failed, key=lambda f: abs((seconds(f[6]) or t) - t))
    running = [f for f in recs if 0 < seconds(f[5]) <= t and (not seconds(f[6]) or seconds(f[6]) >= t)]
    return max(running, key=lambda f: seconds(f[5])) if running else None


def node_metrics():
    """Each node's state and the latest job started on it."""
    state, out = {}, []
    try:
        for line in run("sinfo", "-N", "-h", "-o", "%N|%T|%E|%P|%H").splitlines():
            node, st, reason, part, since = (line.split("|") + ["", "", "", ""])[:5]
            state.setdefault(node, (st, "" if reason == "none" else reason, part.rstrip("*"), seconds(since)))   # a line per partition
        last = last_jobs(state, RECENT_DAYS)
        if missing := [n for n in state if n not in last]:
            last.update(last_jobs(state, LOOKBACK_DAYS, missing))
    except Exception:   # noqa: BLE001  sinfo or sacct missing or hung: push the jobs without the nodes
        return []
    for node, (st, reason, part, since) in sorted(state.items()):
        out.append(f'slurm_node_state{{instance="{node}",state="{esc(st)}",reason="{esc(reason)}",partition="{esc(part)}"}} 1')
        if since and DOWN.search(st):
            out.append(f'slurm_node_reason_seconds{{instance="{node}"}} {since:.0f}')
            try:
                f = cause_job(node, since)
            except Exception:   # noqa: BLE001  sacct hung: no cause this round
                f = None
            if f:
                out.append(f'slurm_node_cause_job_seconds{{instance="{node}",cause_job="{esc(f[0])}",cause_user="{esc(f[1])}",'
                           f'cause_name="{esc(f[2])}",cause_state="{esc(f[4].split()[0])}"}} {seconds(f[6]) or seconds(f[5]):.0f}')
    for node, f in sorted(last.items()):
        lab = (f'instance="{node}",job_id="{esc(f[0])}",user="{esc(f[1])}",job_name="{esc(f[2])}",partition="{esc(f[3])}",'
               f'job_state="{esc(f[4].split()[0])}"')   # 'CANCELLED by 123' -> CANCELLED
        out += [f"slurm_node_last_job_start_seconds{{{lab}}} {seconds(f[5]):.0f}",
                f"slurm_node_last_job_end_seconds{{{lab}}} {seconds(f[6]):.0f}"]
    return out


def metrics():
    gpus, cpus, alloc, pending = [], [], [], []
    ok = 1
    try:
        text = run("scontrol", "show", "job", "-d", "--oneliner")
    except Exception:   # noqa: BLE001  scontrol missing or hung: report it, keep pushing
        text, ok = "", 0
    for job in text.splitlines():
        f = dict(re.findall(r"(\w+)=(\S*)", job))
        jl = ",".join(f'{k}="{esc(v)}"' for k, v in dict(job_id=f.get("JobId", ""), user=f.get("UserId", "").split("(")[0],
                      job_name=f.get("JobName", ""), partition=f.get("Partition", "")).items())
        if f.get("JobState") == "PENDING":
            gres = re.search(r"gres/gpu(?::[^=,]+)?=(\d+)", f.get("ReqTRES", "") or f.get("TRES", ""))
            pending.append((f'{jl},reason="{esc(f.get("Reason", ""))}"', seconds(f.get("SubmitTime", "")),
                            int(gres.group(1)) if gres else 0, int(f.get("NumNodes", "1").split("-")[0] or 1), int(f.get("Priority", "0") or 0)))
        if f.get("JobState") != "RUNNING":
            continue
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
    used, now = rss([j for j, *_ in alloc]), time.time()
    lines = ["# HELP slurm_job_gpu 1 for each GPU a running Slurm job holds.", "# TYPE slurm_job_gpu gauge", *gpus,
             "# HELP slurm_job_cpu 1 for each CPU core a running Slurm job is pinned to.", "# TYPE slurm_job_cpu gauge", *cpus,
             "# HELP slurm_job_cpus CPUs allocated to the job.", "# TYPE slurm_job_cpus gauge",
             *[f"slurm_job_cpus{{{jl}}} {n}" for _, jl, n, _ in alloc],
             "# HELP slurm_job_memory_bytes Memory allocated to the job (0: none requested).", "# TYPE slurm_job_memory_bytes gauge",
             *[f"slurm_job_memory_bytes{{{jl}}} {m:.0f}" for _, jl, _, m in alloc],
             "# HELP slurm_job_rss_bytes Memory in use by the job's running steps (sstat).", "# TYPE slurm_job_rss_bytes gauge",
             *[f"slurm_job_rss_bytes{{{jl}}} {used[j]:.0f}" for j, jl, _, _ in alloc if j in used],
             "# HELP slurm_job_gpu_up 1 if scontrol answered.", "# TYPE slurm_job_gpu_up gauge", f"slurm_job_gpu_up {ok}",
             "# HELP slurm_job_pending_seconds How long a pending job has waited since it was submitted.", "# TYPE slurm_job_pending_seconds gauge",
             *[f"slurm_job_pending_seconds{{{pl}}} {max(0, now - sub):.0f}" for pl, sub, *_ in pending if sub],
             "# HELP slurm_job_pending_gpus GPUs a pending job asks for.", "# TYPE slurm_job_pending_gpus gauge",
             *[f"slurm_job_pending_gpus{{{pl}}} {g}" for pl, _, g, _, _ in pending],
             "# HELP slurm_job_pending_nodes Nodes a pending job asks for.", "# TYPE slurm_job_pending_nodes gauge",
             *[f"slurm_job_pending_nodes{{{pl}}} {n}" for pl, _, _, n, _ in pending],
             "# HELP slurm_job_pending_priority A pending job's priority (0: held).", "# TYPE slurm_job_pending_priority gauge",
             *[f"slurm_job_pending_priority{{{pl}}} {pr}" for pl, *_, pr in pending]]
    nodes = node_metrics()
    lines += ["# HELP slurm_node_state 1 for each node's Slurm state (sinfo).", "# TYPE slurm_node_state gauge",
              *[n for n in nodes if n.startswith("slurm_node_state")],
              "# HELP slurm_node_last_job_start_seconds When the latest job on the node started.",
              "# TYPE slurm_node_last_job_start_seconds gauge", *[n for n in nodes if n.startswith("slurm_node_last_job_start")],
              "# HELP slurm_node_last_job_end_seconds When it ended (0 while it runs).",
              "# TYPE slurm_node_last_job_end_seconds gauge", *[n for n in nodes if n.startswith("slurm_node_last_job_end")],
              "# HELP slurm_node_reason_seconds When Slurm set a down or drained node's reason.",
              "# TYPE slurm_node_reason_seconds gauge", *[n for n in nodes if n.startswith("slurm_node_reason")],
              "# HELP slurm_node_cause_job_seconds The job that took the node down, and when it ended or started.",
              "# TYPE slurm_node_cause_job_seconds gauge", *[n for n in nodes if n.startswith("slurm_node_cause")]]
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
