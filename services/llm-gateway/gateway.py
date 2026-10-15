"""The LLM gateway: one OpenAI-compatible endpoint for Dynamo's models, with access tokens, a critical-traffic flag and
on-demand model switching (only one model is awake on the GPUs at a time; the others sleep with their weights dropped
and reload them from disk on waking: vLLM's level-1 sleep, which keeps them in CPU memory, garbled output after a wake
and ran the trays out of CPU memory).

    POST /v1/chat/completions, /v1/completions, /v1/responses, ...   forwarded to the frontend of the body's `model`
    GET  /v1/models    every configured model, awake or asleep        GET /health, GET /metrics (Prometheus)
    GET  /metrics/<model>/frontend|engine   that model's Dynamo metrics, relayed from 127.0.0.1 (Prometheus job dynamo)

Tokens: `Authorization: Bearer <LLM_TOKEN>` (normal) or `<LLM_CRITICAL_TOKEN>`. A request is critical when it carries
`X-Priority: critical` *and* the critical token; the flag from any other token is ignored. Critical requests:
  - are scheduled ahead of normal ones inside the engine (x-dynamo-request-priority, vLLM's priority scheduler);
  - for a sleeping model, switch to it at once (the awake model finishes what it is running, then sleeps).
A normal request for a sleeping model waits until the awake model has been idle IDLE_S, or until it has waited
MAX_WAIT_S, so normal traffic can't keep the GPUs swapping, and no model starves.

Config: GATEWAY_CONFIG (JSON: models {name: {frontend, system, model_path}}, default, order), GATEWAY_LISTEN host:port.
"""
import asyncio
import json
import os
import time

from aiohttp import ClientSession, ClientTimeout, web

CFG = json.load(open(os.environ.get("GATEWAY_CONFIG", "/config/gateway.json")))
MODELS = CFG["models"]
TOKENS = {os.environ.get("LLM_TOKEN", ""): "normal", os.environ.get("LLM_CRITICAL_TOKEN", ""): "critical"}
TOKENS.pop("", None)
IDLE_S = float(os.environ.get("IDLE_S", 60))            # idle time before a normal request may switch models; untuned
MAX_WAIT_S = float(os.environ.get("MAX_WAIT_S", 600))   # a waiting normal request switches anyway after this; untuned
WAKE_TIMEOUT_S = 1200   # a wake: reloading the weights from disk (GLM-5.3's 704 GiB took 163 s; Kimi K3 is 2x) and listing
PRIORITY = {"critical": "100", "normal": "0"}   # Dynamo's x-dynamo-request-priority: higher runs first

state = {"awake": None, "switching": None, "switches": 0}
inflight = {m: {"normal": 0, "critical": 0} for m in MODELS}
last_done = {m: time.monotonic() for m in MODELS}
waiting = {"normal": 0, "critical": 0}
counts = {}
lock = asyncio.Lock()
http = None


async def listed(name):
    """True when the model's frontend lists it, i.e. its engine is awake and registered."""
    try:
        async with http.get(MODELS[name]["frontend"] + "/v1/models", timeout=ClientTimeout(total=5)) as r:
            return any(m.get("id") == name for m in (await r.json()).get("data", []))
    except Exception:   # noqa: BLE001  a frontend that doesn't answer lists nothing
        return False


async def control(name, action, body):
    """POST to the engine's system API (`control/sleep`, `update_weights_from_disk`, ...); it answers errors with 200."""
    async with http.post(f"{MODELS[name]['system']}/engine/{action}", json=body,
                         timeout=ClientTimeout(total=WAKE_TIMEOUT_S)) as r:
        text = await r.text()
        try:
            ok = r.status < 300 and json.loads(text).get("status") == "ok"
        except ValueError:
            ok = False
        if not ok:
            raise web.HTTPServiceUnavailable(text=f"{action} {name}: HTTP {r.status} {text}")


async def sleep(name):
    await control(name, "control/sleep", {"level": 2})   # level 2: drop the weights too; nothing kept in CPU memory


async def wake(name):
    """Wake `name` and reload its weights from disk (a level-2 sleep left the weights' memory unset)."""
    await control(name, "control/wake_up", {})
    await control(name, "pause_generation", {"mode": "abort", "clear_cache": True})
    await control(name, "update_weights_from_disk", {"model_path": MODELS[name]["model_path"]})
    await control(name, "resume_generation", {})


async def switch_to(name):
    """Put the awake model to sleep and wake `name` (callers hold `lock`)."""
    cur = state["awake"]
    if cur is None and await listed(name):   # came up after the gateway started (ctl.sh dynamo up starts the gateway first)
        state["awake"] = name
        return
    state["switching"] = name
    try:
        if cur and cur != name:
            await sleep(cur)   # drains what it is running first
            state["awake"] = None
        await wake(name)
        deadline = time.monotonic() + WAKE_TIMEOUT_S
        while not await listed(name):
            if time.monotonic() > deadline:
                raise web.HTTPServiceUnavailable(text=f"{name} did not come back after waking")
            await asyncio.sleep(2)
        state["awake"] = name
        state["switches"] += 1
    finally:
        state["switching"] = None


async def ensure_awake(name, cls):
    if state["awake"] == name:
        return
    waiting[cls] += 1
    start = time.monotonic()
    try:
        while state["awake"] != name:
            cur = state["awake"]
            idle = cur is None or (sum(inflight[cur].values()) == 0 and time.monotonic() - last_done[cur] >= IDLE_S)
            busy_critical = cur is not None and inflight[cur]["critical"] > 0
            if cls == "critical" or (not busy_critical and (idle or time.monotonic() - start >= MAX_WAIT_S)):
                async with lock:
                    if state["awake"] != name:
                        await switch_to(name)
                return
            await asyncio.sleep(1)
    finally:
        waiting[cls] -= 1


def classify(request):
    auth = request.headers.get("Authorization", "")
    kind = TOKENS.get(auth.removeprefix("Bearer ").strip())
    if kind is None:
        raise web.HTTPUnauthorized(text="LLM gateway: send Authorization: Bearer <token>")
    flagged = request.headers.get("X-Priority", "").lower() == "critical"
    return "critical" if flagged and kind == "critical" else "normal"


async def forward(request):
    cls = classify(request)
    body = await request.read()
    try:
        name = json.loads(body or b"{}").get("model")
    except ValueError:
        raise web.HTTPBadRequest(text="body is not JSON")
    if name not in MODELS:
        raise web.HTTPNotFound(text=f"unknown model {name!r}; GET /v1/models lists them")
    counts[(name, cls)] = counts.get((name, cls), 0) + 1
    await ensure_awake(name, cls)
    inflight[name][cls] += 1
    try:
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("authorization", "host", "content-length", "x-priority",
                                        "x-dynamo-request-priority", "x-dynamo-request-strict-priority")}
        headers["x-dynamo-request-priority"] = PRIORITY[cls]
        async with http.request(request.method, MODELS[name]["frontend"] + request.path_qs, data=body, headers=headers,
                                timeout=ClientTimeout(total=None, sock_read=3600)) as r:
            resp = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items()
                                                                if k.lower() in ("content-type", "cache-control")})
            await resp.prepare(request)
            async for chunk in r.content.iter_any():   # streams (SSE) pass through as they come
                await resp.write(chunk)
            await resp.write_eof()
            return resp
    finally:
        inflight[name][cls] -= 1
        last_done[name] = time.monotonic()


async def models(request):
    classify(request)
    return web.json_response({"object": "list", "data": [
        {"id": n, "object": "model", "owned_by": "flywheel", "awake": n == state["awake"]} for n in CFG["order"]]})


async def health(_):
    return web.json_response({"awake": state["awake"], "switching": state["switching"]})


async def metrics(_):
    lines = ["# TYPE llm_gateway_awake gauge"] + [f'llm_gateway_awake{{model="{n}"}} {int(n == state["awake"])}' for n in MODELS]
    lines += ["# TYPE llm_gateway_switches_total counter", f"llm_gateway_switches_total {state['switches']}",
              "# TYPE llm_gateway_inflight gauge"]
    lines += [f'llm_gateway_inflight{{model="{n}",class="{c}"}} {v}' for n, d in inflight.items() for c, v in d.items()]
    lines += ["# TYPE llm_gateway_waiting gauge"] + [f'llm_gateway_waiting{{class="{c}"}} {v}' for c, v in waiting.items()]
    lines += ["# TYPE llm_gateway_requests_total counter"]
    lines += [f'llm_gateway_requests_total{{model="{n}",class="{c}"}} {v}' for (n, c), v in counts.items()]
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


async def model_metrics(request):
    """A model's Dynamo metrics for Prometheus, which can't reach them on 127.0.0.1: its frontend's (requests, tokens,
    time to first token, inter-token latency) or its engine's (vLLM: tokens, running and waiting requests, KV cache).
    Read-only; empty while the engine sleeps."""
    name, part = request.match_info["model"], request.match_info["part"]
    if name not in MODELS or part not in ("frontend", "engine"):
        raise web.HTTPNotFound()
    url = MODELS[name]["frontend" if part == "frontend" else "system"] + "/metrics"
    try:
        async with http.get(url, timeout=ClientTimeout(total=10)) as r:
            return web.Response(text=await r.text() if r.status == 200 else "", content_type="text/plain")
    except Exception:   # noqa: BLE001  a stopped frontend or engine has no metrics; the scrape still succeeds
        return web.Response(text="", content_type="text/plain")


async def startup(app):
    global http
    http = ClientSession()
    for n in CFG["order"]:   # whichever engine is already awake (after a gateway restart)
        if await listed(n):
            state["awake"] = n
            break


async def cleanup(app):
    await http.close()


app = web.Application(client_max_size=64 * 2 ** 20)   # 64 MiB bodies: a few images as data: URLs; untuned
app.router.add_get("/v1/models", models)
app.router.add_get("/health", health)
app.router.add_get("/metrics", metrics)
app.router.add_get("/metrics/{model}/{part}", model_metrics)
app.router.add_route("POST", "/v1/{tail:.*}", forward)
app.on_startup.append(startup)
app.on_cleanup.append(cleanup)
host, _, port = os.environ.get("GATEWAY_LISTEN", "127.0.0.1:8199").rpartition(":")
web.run_app(app, host=host, port=int(port), access_log=None)
