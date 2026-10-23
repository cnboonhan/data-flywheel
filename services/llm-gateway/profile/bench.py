"""Streaming latency of one model through one path: python3 bench.py <label> <base url> <model> [CA file].

Sends the same streamed chat request N times and prints one row: the median time to first token, total time and decode
speed (completion tokens over the time after the first token), of all but the first request, which warms caches.
Standard library only, so it runs on any node.
"""
import json
import ssl
import statistics
import sys
import time
import urllib.request

label, base, model = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3]
ctx = ssl.create_default_context(cafile=sys.argv[4]) if len(sys.argv) > 4 else None
N = 6               # requests per path: 1 warm-up + 5 timed; the median of 5 held within 1-2 ms between repeats
MAX_TOKENS = 128    # long enough to time decoding (~0.4 s at 300 tok/s), short enough to keep a run to seconds
TIMEOUT_S = 300     # a model that has to start, or a long queue, fails the row instead of hanging the test
BODY = {"model": model, "stream": True, "max_tokens": MAX_TOKENS, "temperature": 0,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Count from 1 to 40, separated by commas."}],
        "chat_template_kwargs": {"enable_thinking": False}}   # reasoning models answer at once; others ignore it

rows = []
try:
    for i in range(N):
        t0, first, toks = time.perf_counter(), None, 0
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(BODY).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=ctx, timeout=TIMEOUT_S) as r:
            for line in r:
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("usage"):
                    toks = ev["usage"]["completion_tokens"]
                for c in ev.get("choices", []):
                    d = c.get("delta", {})
                    if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                        first = time.perf_counter()
        end = time.perf_counter()
        if i and first is not None:
            rows.append((first - t0, end - t0, toks / max(end - first, 1e-6)))
except Exception as e:   # noqa: BLE001  one failed path shouldn't hide the others' rows
    print(f"| {label} | failed: {type(e).__name__}: {str(e)[:80]} | | |")
    sys.exit(0)

med = lambda k: statistics.median(r[k] for r in rows)
print(f"| {label} | {med(0) * 1000:.0f} ms | {med(1):.2f} s | {med(2):.0f} tok/s |")
