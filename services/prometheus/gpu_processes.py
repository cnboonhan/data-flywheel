"""Who uses each GPU, and each node's CPU memory, for Prometheus: what the cluster's exporters can't say.

    python3 gpu_processes.py            (ctl.sh monitor; its timer runs it every minute on SERVICE_NODE)

Over ssh (read-only) on every node in nodes.json, at once: nvidia-smi's processes per GPU, each process's user (for a root-owned container, whoever's `docker run` is still attached to it, when that is clear),
its workload (its Docker container's name, else its Slurm job, else its script or command) and the memory it holds on
the GPU; and the node's CPU memory alone (the NUMA nodes with CPUs: on a GB300 node the GPUs' memory is system memory
too, so node_exporter's MemTotal and MemAvailable count it). Pushes to the Pushgateway (Caddy's /pushgateway route,
PUSHGATEWAY_TOKEN), replacing the last push, so finished processes disappear:

    gpu_process_memory_bytes{node, gpu, user, workload, slurm_job}   GPU memory the processes hold (sum)
    gpu_process_count{node, gpu, user, workload, slurm_job}
    node_cpu_memory_total_bytes{node}, node_cpu_memory_available_bytes{node}   (available: free + file cache + reclaimable slab)
    gpu_processes_node_up{node}                                      1 when the node answered (0: unknown, e.g. ssh not allowed)

Env: SERVICE_HOST, CADDY_PORT, PUSHGATEWAY_TOKEN (services/.env), and the CA (FLYWHEEL_CA, else AWS_CA_BUNDLE).
"""
import concurrent.futures
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.request

here = os.path.dirname(os.path.abspath(__file__))
SSH_TIMEOUT_S = 40   # per node: a hung node mustn't hold up the minute's push; untuned

# Runs on each node: GPUs, processes (pid, bus, MiB), per process its user, cgroup and command line, Docker names,
# and the CPU NUMA nodes' meminfo. One line per record, tab-separated, tagged by its first field.
REMOTE = r'''
nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader 2>/dev/null | sed 's/^/G\t/; s/, /\t/'
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits 2>/dev/null | sed 's/^/P\t/; s/, /\t/g' |
while IFS=$'\t' read -r tag bus pid mib; do
  user=$(ps -o user:64= -p "$pid" 2>/dev/null); cg=$(tr '\n' ' ' < /proc/$pid/cgroup 2>/dev/null)
  cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-300)
  printf 'P\t%s\t%s\t%s\t%s\t%s\t%s\n' "$bus" "$pid" "$mib" "${user:-?}" "$cg" "$cmd"
done
(docker ps --no-trunc --format '{{.ID}} {{.Names}} {{.Image}}' 2>/dev/null || bash -lc "docker ps --no-trunc --format '{{.ID}} {{.Names}} {{.Image}}'" 2>/dev/null) | sed 's/^/D\t/; s/ /\t/g'
ps -eo user:64=,args= 2>/dev/null | awk '$2 == "docker" && ($3 == "run" || ($3 == "container" && $4 == "run")) {u = $1; $1 = ""; printf "R\t%s\t%s\n", u, $0}'
for d in /sys/devices/system/node/node[0-9]*; do
  [ -n "$(cat $d/cpulist)" ] && awk -v n="${d##*node}" '/MemTotal|MemFree|Active\(file\)|Inactive\(file\)|SReclaimable/ {printf "M\t%s\t%s\t%s\n", n, $3, $4}' $d/meminfo
done
'''


def workload(cg, cmd, docker):
    """The container's name, else Slurm job, else the script a python runs, else the command's name."""
    m = re.search(r"docker[-/]([0-9a-f]{64})", cg)
    if m and m.group(1) in docker:
        return docker[m.group(1)], ""
    job = re.search(r"job_(\d+)", cg)
    args = cmd.split()
    name = os.path.basename(args[0]) if args else "?"
    if "python" in name:   # the module (-m) or script it runs
        if "-m" in args[1:-1]:
            name = args[args.index("-m") + 1]
        else:
            name = os.path.basename(next((a for a in args[1:] if a.endswith(".py")), "")) or name
    return name.split(":")[0] if name.startswith("VLLM::") else name, job.group(1) if job else ""


def launcher(user, cg, docker, runs):
    """A container's processes run as its image's user (root for most), not whoever ran it: when exactly one `docker run`
    client still attached on the node names the container's image, that client's user, marked as such."""
    m = re.search(r"docker[-/]([0-9a-f]{64})", cg)
    if user != "root" or not m or m.group(1) not in docker:
        return user
    image = docker[m.group(1)][1]
    users = {u for u, args in runs if image in args.split() or image.removesuffix(":latest") in args.split()}
    return f"{users.pop()} (docker run)" if len(users) == 1 else user


def collect(node, ip):
    try:
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", ip, "bash", "-s"], input=REMOTE,
                           capture_output=True, text=True, timeout=SSH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return node, None
    if r.returncode == 255 or not any(l.startswith("M\t") for l in r.stdout.splitlines()):   # ssh refused or failed
        return node, None
    out = r.stdout
    gpus, procs, docker, mem, runs = {}, [], {}, {}, []
    for line in out.splitlines():
        f = line.split("\t")
        if f[0] == "G" and len(f) >= 3:
            gpus[f[2].strip().lower()[-12:]] = f[1].strip()   # bus id, short form (nvidia-smi pads domains differently)
        elif f[0] == "P" and len(f) >= 7:
            procs.append(f[1:])
        elif f[0] == "D" and len(f) >= 4:
            docker[f[1]] = (f[2], f[3])
        elif f[0] == "R" and len(f) >= 3:
            runs.append((f[1], f[2]))
        elif f[0] == "M" and len(f) >= 4:
            mem[f[2]] = mem.get(f[2], 0) + int(f[3]) * 1024
    rows = {}
    for bus, _pid, mib, user, cg, cmd in procs:
        gpu = gpus.get(bus.strip().lower()[-12:], "?")
        wl, job = workload(cg, cmd, {k: v[0] for k, v in docker.items()})
        user = launcher(user, cg, docker, runs)
        key = (gpu, user, wl, job)
        b, c = rows.get(key, (0, 0))
        rows[key] = (b + float(mib or 0) * 2**20, c + 1)
    return node, {"rows": rows, "mem": mem}


def esc(v):
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def main():
    nodes = [(t["labels"]["node"], t["targets"][0]) for t in json.load(open(os.path.join(here, "nodes.json")))]
    lines = ["# TYPE gpu_process_memory_bytes gauge", "# TYPE gpu_process_count gauge",
             "# TYPE node_cpu_memory_total_bytes gauge", "# TYPE node_cpu_memory_available_bytes gauge",
             "# TYPE gpu_processes_node_up gauge"]
    with concurrent.futures.ThreadPoolExecutor(16) as pool:   # 16 nodes at a time; untuned
        for node, r in pool.map(lambda n: collect(*n), nodes):
            lines.append(f'gpu_processes_node_up{{node="{node}"}} {int(r is not None)}')
            if r is None:
                continue
            for (gpu, user, wl, job), (b, c) in r["rows"].items():
                lab = f'node="{node}",gpu="{gpu}",user="{esc(user)}",workload="{esc(wl)}",slurm_job="{job}"'
                lines += [f"gpu_process_memory_bytes{{{lab}}} {b:.0f}", f"gpu_process_count{{{lab}}} {c}"]
            m = r["mem"]
            if m.get("MemTotal:"):
                avail = m.get("MemFree:", 0) + m.get("Active(file):", 0) + m.get("Inactive(file):", 0) + m.get("SReclaimable:", 0)
                lines += [f'node_cpu_memory_total_bytes{{node="{node}"}} {m["MemTotal:"]}',
                          f'node_cpu_memory_available_bytes{{node="{node}"}} {avail}']
    body = ("\n".join(lines) + "\n").encode()
    if "--print" in sys.argv:
        sys.stdout.write(body.decode())
        return
    url = f"https://{os.environ['SERVICE_HOST']}:{os.environ['CADDY_PORT']}/pushgateway/metrics/job/gpu_processes"
    ca = os.environ.get("FLYWHEEL_CA") or os.environ.get("AWS_CA_BUNDLE")
    req = urllib.request.Request(url, body, method="PUT", headers={"Authorization": "Bearer " + os.environ["PUSHGATEWAY_TOKEN"]})
    urllib.request.urlopen(req, context=ssl.create_default_context(cafile=ca), timeout=30).read()


if __name__ == "__main__":
    main()
