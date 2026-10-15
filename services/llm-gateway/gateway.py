"""The LLM gateway: one OpenAI-compatible endpoint for Dynamo's served model, with access tokens and a critical-traffic
flag. Which model is served is decided in MLflow, not by requests: the registered model whose alias `dynamo` is set
(services/dynamo/README.md); moving the alias swaps the model (ctl.sh dynamo sync restarts the engines).

    POST /v1/chat/completions, /v1/completions, /v1/responses, ...   forwarded to the frontend of the body's `model`
    GET  /v1/models    the served models: ready, context window, licence, inputs        GET /health, GET /metrics
    GET  /metrics/<model>/frontend|engine   that model's Dynamo metrics, relayed from 127.0.0.1 (Prometheus job dynamo)

No token is needed. A request is critical when it carries `X-Priority: critical` *and*
`Authorization: Bearer <LLM_CRITICAL_TOKEN>` (the flag without that token is ignored): it is scheduled ahead of normal
ones inside the engine (x-dynamo-request-priority, vLLM's priority scheduler).

Each node that serves models runs a gateway; the one on DYNAMO_HEAD (Caddy's llm.<host>) also forwards the other nodes'
models to their gateway, which classifies and counts the request.

Config: GATEWAY_CONFIG (JSON: models {name: {frontend, system} served here, or {gateway} on another node's gateway},
known [names in models.yaml]), GATEWAY_LISTEN host:port.
"""
import json
import os

from aiohttp import ClientSession, ClientTimeout, web

CFG = json.load(open(os.environ.get("GATEWAY_CONFIG", "/config/gateway.json")))
MODELS = CFG["models"]
CRITICAL_TOKEN = os.environ.get("LLM_CRITICAL_TOKEN", "")
PRIORITY = {"critical": "100", "normal": "0"}   # Dynamo's x-dynamo-request-priority: higher runs first

LOCAL = [m for m, c in MODELS.items() if "frontend" in c]
inflight = {m: {"normal": 0, "critical": 0} for m in LOCAL}
counts = {}
http = None


async def listed(name):
    """True when the model's frontend lists it, i.e. its engine is up and registered (for another node's model, its
    gateway says so)."""
    try:
        if "gateway" in MODELS[name]:
            async with http.get(MODELS[name]["gateway"] + "/health", timeout=ClientTimeout(total=5)) as r:
                return bool((await r.json()).get("served", {}).get(name))
        async with http.get(MODELS[name]["frontend"] + "/v1/models", timeout=ClientTimeout(total=5)) as r:
            return any(m.get("id") == name for m in (await r.json()).get("data", []))
    except Exception:   # noqa: BLE001  a frontend that doesn't answer lists nothing
        return False


def classify(request):
    """critical with the flag and the critical token; anything else (no token, any other token) is normal."""
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    flagged = request.headers.get("X-Priority", "").lower() == "critical"
    return "critical" if flagged and CRITICAL_TOKEN and token == CRITICAL_TOKEN else "normal"


async def forward(request):
    cls = classify(request)
    body = await request.read()
    try:
        name = json.loads(body or b"{}").get("model")
    except ValueError:
        raise web.HTTPBadRequest(text="body is not JSON")
    served = ", ".join(MODELS) or "none"
    if name not in MODELS:
        if name in CFG.get("known", []):
            raise web.HTTPServiceUnavailable(text=f"{name} is not served now (served: {served}); the served model is "
                                                  "chosen by its MLflow alias `dynamo`")
        raise web.HTTPNotFound(text=f"unknown model {name!r} (served: {served})")
    if "gateway" in MODELS[name]:   # another node's: its gateway classifies and counts it
        return await proxy(request, MODELS[name]["gateway"],
                           {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}, body)
    if not await listed(name):
        raise web.HTTPServiceUnavailable(text=f"{name} is starting (minutes after a swap); try again later")
    counts[(name, cls)] = counts.get((name, cls), 0) + 1
    inflight[name][cls] += 1
    try:
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("authorization", "host", "content-length", "x-priority",
                                        "x-dynamo-request-priority", "x-dynamo-request-strict-priority")}
        headers["x-dynamo-request-priority"] = PRIORITY[cls]
        return await proxy(request, MODELS[name]["frontend"], headers, body)
    finally:
        inflight[name][cls] -= 1


async def proxy(request, base, headers, body):
    async with http.request(request.method, base + request.path_qs, data=body, headers=headers,
                            timeout=ClientTimeout(total=None, sock_read=3600)) as r:
        resp = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items()
                                                            if k.lower() in ("content-type", "cache-control")})
        await resp.prepare(request)
        async for chunk in r.content.iter_any():   # streams (SSE) pass through as they come
            await resp.write(chunk)
        await resp.write_eof()
        return resp


async def describe(name):
    """A model's entry in /v1/models: ready, its context window (from its frontend, or another node's gateway) and the
    info models.yaml gives (licence, about, images, reasoning, tools)."""
    entry = {"id": name, "object": "model", "owned_by": "flywheel", "ready": False, "context_window": None,
             **MODELS[name].get("info", {})}
    try:
        url = MODELS[name]["gateway"] if "gateway" in MODELS[name] else MODELS[name]["frontend"]
        async with http.get(url + "/v1/models", timeout=ClientTimeout(total=5)) as r:
            m = next((m for m in (await r.json()).get("data", []) if m.get("id") == name), None)
        if m:
            entry["ready"] = m.get("ready", True)
            entry["context_window"] = m.get("context_window")
    except Exception:   # noqa: BLE001  down or starting: listed as not ready
        pass
    return entry


async def models(request):
    return web.json_response({"object": "list", "data": [await describe(n) for n in MODELS]})


async def health(_):
    return web.json_response({"served": {n: await listed(n) for n in MODELS}})


def local_only(names):
    return [n for n in names if n in LOCAL]   # another node's gateway reports its own models


async def metrics(_):
    lines = ["# TYPE llm_gateway_served gauge"]   # 1 while the model is up and listed, 0 while it starts
    lines += [f'llm_gateway_served{{model="{n}"}} {int(await listed(n))}' for n in local_only(MODELS)]
    lines += ["# TYPE llm_gateway_inflight gauge"]
    lines += [f'llm_gateway_inflight{{model="{n}",class="{c}"}} {v}' for n, d in inflight.items() for c, v in d.items()]
    lines += ["# TYPE llm_gateway_requests_total counter"]
    lines += [f'llm_gateway_requests_total{{model="{n}",class="{c}"}} {v}' for (n, c), v in counts.items()]
    return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")


async def model_metrics(request):
    """A model's Dynamo metrics for Prometheus, which can't reach them on 127.0.0.1: its frontend's (requests, tokens,
    time to first token, inter-token latency) or its engine's (vLLM: tokens, running and waiting requests, KV cache).
    Read-only; empty while the engine starts."""
    name, part = request.match_info["model"], request.match_info["part"]
    if name not in LOCAL or part not in ("frontend", "engine"):
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
