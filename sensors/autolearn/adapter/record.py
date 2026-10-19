"""Record the benchmark's observation/action stream, one frame per simulator tick, as raw episode folders.

An episode folder holds the camera images as the benchmark encoded them (`<camera>/<frame>.jpg`), one JSON line per
tick in `frames.jsonl` (the joint readings the benchmark reported and the absolute joint targets the adapter sent
for that tick) and `episode.json` (instruction, task, outcome). `to_lerobot.py` turns a set of them into a LeRobot
v2.1 dataset; keeping the raw folders means a format change never costs a simulator run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

STATE_KEYS = ("arm_joint_states", "gripper_states", "waist_joint_states", "head_joint_states")
ACTION_KEYS = ("left_arm", "right_arm", "left_effector", "right_effector", "waist", "head")


def reply_row(reply: dict, i: int) -> dict:
    """The i-th tick of a multi-row reply as a one-row reply (every per-joint-group list sliced)."""
    out = {}
    for k, v in reply.items():
        if isinstance(v, dict) and "values" in v:
            out[k] = {**v, "values": [v["values"][i]]}
        elif isinstance(v, list) and v and isinstance(v[0], list):
            out[k] = [v[i]]
        else:
            out[k] = v
    return out


class EpisodeRecorder:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.dir: Path | None = None
        self._frames = None
        self._n = 0

    def start(self, meta: dict) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        n = len([p for p in self.root.iterdir() if p.is_dir()])
        self.dir = self.root / f"episode_{n:04d}"
        self.dir.mkdir()
        self._frames = (self.dir / "frames.jsonl").open("w")
        self._n = 0
        (self.dir / "episode.json").write_text(json.dumps({**meta, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=1))
        return self.dir

    def add(self, raw: dict, reply: dict) -> None:
        """One tick: the observation `raw` (as received) and the one-row `reply` that answers it."""
        if self.dir is None:
            return
        for key, img in raw.get("images", {}).items():
            if key.endswith("_depth") or img is None:
                continue
            (self.dir / key).mkdir(exist_ok=True)
            data = img["image_data"]
            ext = "jpg" if data[:2] == b"\xff\xd8" else "png"
            (self.dir / key / f"{self._n:05d}.{ext}").write_bytes(data)
        st = raw.get("states", {})
        action = {}
        for k in ACTION_KEYS:
            v = reply.get(k)
            if isinstance(v, dict):
                action[k] = v["values"][0]
            elif isinstance(v, list):
                action[k] = v[0]
        self._frames.write(json.dumps({"frame": self._n, "state": {k: st[k] for k in STATE_KEYS if k in st},
                                       "end_pose": st.get("end_pose"), "action": action}) + "\n")
        self._n += 1

    def finish(self, outcome: dict) -> int:
        if self.dir is None:
            return 0
        self._frames.close()
        meta = json.loads((self.dir / "episode.json").read_text())
        (self.dir / "episode.json").write_text(json.dumps({**meta, **outcome, "frames": self._n}, indent=1))
        n, self.dir, self._frames = self._n, None, None
        return n
