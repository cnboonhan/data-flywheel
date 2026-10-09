"""Robot descriptions in services/slurm/embodiments/<name>.yaml.

    from embodiment import load
    emb = load("arx_x5"); emb.action_dim("joint")  # 14

    python embodiment.py arx_x5     # shell assignments for the sbatch jobs: EMB_DATA_ENV_CFG=..., EMB_XPL_ENV_CFG=...
"""

import dataclasses
import os
import shlex
import sys

import yaml

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embodiments")

# xspark keys per action type, per arm, concatenated over arms in order.
ACTION_KEYS = {"joint": ["arm_joint_states", "ee_joint_states"], "ee": ["ee_poses", "ee_joint_states"]}


@dataclasses.dataclass
class Embodiment:
    name: str
    arms: list
    arm_dim: int
    ee_dim: int
    cameras: list
    data_env_cfg: str
    xpolicylab: dict

    def keys(self, action):
        return [f"{arm}_{k}" for arm in self.arms for k in ACTION_KEYS[action]]

    def action_dim(self, action):
        per_arm = (self.arm_dim if action == "joint" else 7) + self.ee_dim
        return per_arm * len(self.arms)


def load(name):
    path = os.path.join(DIR, f"{name}.yaml")
    if name.startswith("_") or not os.path.isfile(path):
        known = sorted(f[:-5] for f in os.listdir(DIR) if f.endswith(".yaml") and not f.startswith("_"))
        raise SystemExit(f"unknown embodiment {name!r}; known: {', '.join(known)} (add embodiments/{name}.yaml)")
    d = yaml.safe_load(open(path))
    d.setdefault("xpolicylab", {})
    return Embodiment(**{f.name: d[f.name] for f in dataclasses.fields(Embodiment)})


if __name__ == "__main__":
    e = load(sys.argv[1])
    for k, v in {"EMB_NAME": e.name, "EMB_DATA_ENV_CFG": e.data_env_cfg,
                 "EMB_XPL_ENV_CFG": e.xpolicylab.get("env_cfg") or "", "EMB_EVAL_ENV_CFG": e.xpolicylab.get("eval_env_cfg") or ""}.items():
        print(f"{k}={shlex.quote(str(v))}")
