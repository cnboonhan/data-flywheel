"""A Genie Sim policy server that holds still: the first probe of the benchmark's inference protocol.

    python3 probe_server.py [--port 8999] [--horizon 30] [--dump obs.pkl]

Answers every `infer` request with a chunk that repeats the observed joints, so `geniesim benchmark run <config>
--infer-host=<host>:<port>` runs an episode end to end without a policy; saves the first request (minus image
bytes) to --dump and prints what each request carried. The protocol is geniesim_benchmark's
`benchmark/policy/corobotpolicy.py` (msgpack over WebSocket; the reply packed with plain lists, never numpy).
"""

import argparse
import asyncio
import pickle
import time

import msgpack
import numpy as np
import websockets

try:
    import msgpack_numpy
except ImportError:                       # the container has geniesim_benchmark's copy, not the pip package
    from geniesim_benchmark.utils import msgpack_numpy

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=8999)
p.add_argument("--horizon", type=int, default=30)
p.add_argument("--dump", default="")
args = p.parse_args()
count = 0


def summary(params):
    img = {k: (v.get("encoding"), v.get("width"), v.get("height"), len(v.get("image_data", b"")))
           for k, v in params.get("images", {}).items()}
    st = {k: (len(v) if isinstance(v, (list, tuple)) else type(v).__name__) for k, v in params.get("states", {}).items()}
    prog = [(t.get("name"), t.get("score"), t.get("status")) for t in params.get("task_progress", [])]
    return f"episode {params.get('episode_idx')} done={params.get('episode_done')} prompt={params.get('prompt')!r}\n" \
           f"  images {img}\n  states {st}\n  progress {prog}"


async def handle(ws):
    global count
    async for raw in ws:
        req = msgpack_numpy.unpackb(raw, raw=False)
        params = req.get("params", {})
        count += 1
        print(f"[{time.strftime('%H:%M:%S')}] request {count}: {summary(params)}", flush=True)
        if count == 1 and args.dump:
            slim = {**params, "images": {k: {kk: vv for kk, vv in v.items() if kk != "image_data"}
                                         for k, v in params.get("images", {}).items()}}
            with open(args.dump, "wb") as f:
                pickle.dump(slim, f)
        st = params.get("states", {})
        arm = np.asarray(st.get("arm_joint_states", []), dtype=float).tolist()
        grip = np.asarray(st.get("gripper_states", [0, 0]), dtype=float).tolist()
        waist = np.asarray(st.get("waist_joint_states", []), dtype=float).tolist()
        half = len(arm) // 2
        H = args.horizon
        result = {
            "left_arm": {"kind": "JOINT_ABS", "values": [arm[:half]] * H},
            "right_arm": {"kind": "JOINT_ABS", "values": [arm[half:]] * H},
            "left_effector": [[grip[0]]] * H,
            "right_effector": [[grip[1]]] * H,
            "need_depth": True,
        }
        if waist:
            result["waist"] = {"kind": "JOINT_ABS", "values": [waist] * H}
        await ws.send(msgpack.packb({"result": result}, use_bin_type=True))


async def main():
    async with websockets.serve(handle, "0.0.0.0", args.port, max_size=None):
        print(f"probe server on :{args.port}, horizon {args.horizon}", flush=True)
        await asyncio.Future()


asyncio.run(main())
