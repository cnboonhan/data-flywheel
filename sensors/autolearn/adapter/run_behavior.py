"""Run an ASPIRE program on BEHAVIOR-1K's R1 Pro with PyRoki IK in place of OmniGibson's cuRobo.

    aspire/.venv-behavior/bin/python adapter/run_behavior.py <program.py> [--task turning_on_radio] [--task-id 0] [--seed 1]

ASPIRE's own BehaviorAdapter drives OmniGibson (reset, cameras, joint targets, success); only the motion providers
differ from `cap-harness run`: IK is `adapter/ik.py` (PyRoki on the R1 Pro URDF, torso pinned) and trajectories come
from ASPIRE's interpolation planner, so nothing depends on OmniGibson's cuRobo R1 Pro embodiment, which it drops on
RTX 50-series GPUs. Base moves still go through the adapter's `servo` planner. Set the OMNIGIBSON_* variables as
for ASPIRE's BEHAVIOR setup (data under sensors/autolearn/behavior-data). Outcomes go to --out as JSON.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

p = argparse.ArgumentParser()
p.add_argument("program", type=Path)
p.add_argument("--task", default="turning_on_radio")
p.add_argument("--task-id", type=int, default=0)
p.add_argument("--seed", type=int, default=1, help="a task instance id present in the dataset")
p.add_argument("--out", type=Path, default=ROOT / "aspire" / "outputs" / "behavior")
p.add_argument("--urdf", type=Path, default=ROOT / "behavior-data/omnigibson-robot-assets/models/r1pro/urdf/r1pro_with_meta_links.urdf")
p.add_argument("--camera", type=int, default=512)
p.add_argument("--horizon", type=int, default=6000)
p.add_argument("--no-providers", action="store_true", help="skip SAM3 and Contact-GraspNet")
args = p.parse_args()

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNIGIBSON_DATA_PATH", str(ROOT / "behavior-data"))
os.environ.setdefault("OMNIGIBSON_APPDATA_PATH", str(ROOT / "aspire" / ".behavior-appdata"))

from cap_harness.api import CapApi  # noqa: E402
from cap_harness.behavior.adapter import BehaviorAdapter  # noqa: E402
from cap_harness.behavior.registry import BehaviorTaskRegistry  # noqa: E402
from cap_harness.contracts import MotionStrategy  # noqa: E402
from cap_harness.registry import BEHAVIOR_PUBLIC_TOOL_NAMES  # noqa: E402
from cap_harness.run import run_program_with_api  # noqa: E402
from ik import UrdfIkProvider  # noqa: E402

registry = BehaviorTaskRegistry()
metadata = registry.resolve(args.task, args.task_id)
adapter = BehaviorAdapter(registry=registry, camera_width=args.camera, camera_height=args.camera, horizon=args.horizon)
out = args.out / time.strftime("%Y%m%dT%H%M%S")
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
obs = adapter.reset(metadata, args.seed)
print(f"reset in {time.time() - t0:.0f} s: {obs.task_context.language!r}, cameras {sorted(obs.cameras)}, "
      f"arms {adapter.arms}", flush=True)


def all_joints():
    robot = adapter.robot
    return dict(zip(list(robot.joints.keys()), np.asarray(robot.get_joint_positions()).tolist()))


ik = UrdfIkProvider(args.urdf, context_source=adapter.get_planning_context, joints_source=all_joints)
ctx = adapter.get_planning_context()
print(f"OmniGibson's R1 Pro: joints {dict(ctx.joint_names)}, end effectors {dict(ctx.end_effector_links)}", flush=True)
missing = [n for names in ctx.joint_names.values() for n in names if n not in ik.names]
for arm, link in ctx.end_effector_links.items():
    if link not in ik.robot.links.names:      # OmniGibson's eef meta link isn't in the URDF: use the gripper link
        side = "left" if arm == "primary" else "right"
        fallback = f"{side}_gripper_link"
        if fallback in ik.robot.links.names:
            ik.link_map[link] = fallback
            print(f"end effector {link!r} not in the URDF; IK targets {fallback!r} instead", flush=True)
        else:
            missing.append(link)
if missing:
    sys.exit(f"the URDF {args.urdf} lacks {missing}: pick the one OmniGibson's R1 Pro names match")
providers = {}
if not args.no_providers:
    from cap_harness.providers.graspnet.client import ContactGraspNetProvider
    from cap_harness.providers.sam3.client import Sam3Provider
    providers = {"segmentation_provider": Sam3Provider(), "grasp_providers": {"contact-graspnet": ContactGraspNetProvider()}}
api = CapApi(adapter, ik_providers={"pyroki": ik}, default_camera="head",
             default_motion_strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation", pose_planner="composed"),
             **providers)
t1 = time.time()
result = run_program_with_api(api=api, program_path=args.program, extension_allowlist=BEHAVIOR_PUBLIC_TOOL_NAMES)
success = bool(adapter.check_success()) if hasattr(adapter, "check_success") else None
record = {"task": args.task, "task_id": args.task_id, "seed": args.seed, "instruction": obs.task_context.language,
          "program_ok": result.ok, "error": None if result.error is None else result.error.message, "result": result.result,
          "calls": len(result.calls), "wall_s": round(time.time() - t1, 1), "success": success, "stdout": result.stdout[-4000:]}
(out / f"{args.task}_{args.seed}.json").write_text(json.dumps(record, indent=1, default=str))
print(f"program_ok={result.ok} success={success} calls={len(result.calls)} {'' if result.error is None else result.error.message}", flush=True)
adapter.close()
