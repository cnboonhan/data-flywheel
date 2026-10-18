"""Run an ASPIRE program against Genie Sim's benchmark.

    aspire/.venv-pyroki/bin/python adapter/run_geniesim.py <program.py> [--config g2op_if_pick_block_color] [--episodes N]

Starts the policy server (port 8999), then waits for `geniesim benchmark run <config> --infer-host=<host>:8999`
(started separately, in the geniesim3 container) to connect. Each benchmark episode runs the program once through
ASPIRE's harness (`run_program_with_api`): SAM3 and Contact-GraspNet from ASPIRE's services on 8114/8115, IK on
PyRoki with the G2 URDF, the interpolation planner. Outcomes go to --out as JSON, one per episode.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from geniesim_adapter import CorobotBridge, G2IkProvider, G2Model, GenieSimAdapter  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("program", type=Path)
p.add_argument("--config", default="g2op_if_pick_block_color", help="the benchmark config name (TaskContext.suite)")
p.add_argument("--port", type=int, default=8999)
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--out", type=Path, default=ROOT / "aspire" / "outputs" / "geniesim")
p.add_argument("--urdf", type=Path, default=ROOT / "GenieSimAssets/robot/curobo_robot/assets/robot/G2/G2_omnipicker_fixed_dual.urdf")
p.add_argument("--usda", type=Path, default=ROOT / "GenieSimAssets/robot/G2_omnipicker/robot_fix.usda")
p.add_argument("--no-providers", action="store_true", help="skip SAM3 and Contact-GraspNet (programs that only move)")
p.add_argument("--debug", action="store_true", help="save the first head image with the grippers projected")
args = p.parse_args()

from cap_harness.api import CapApi  # noqa: E402
from cap_harness.contracts import MotionStrategy  # noqa: E402
from cap_harness.registry import ROBOSUITE_PUBLIC_TOOL_NAMES  # noqa: E402
from cap_harness.run import run_program_with_api  # noqa: E402

model = G2Model(args.urdf, args.usda)
ik = G2IkProvider(model)
bridge = CorobotBridge(args.port)
bridge.start()
adapter = GenieSimAdapter(bridge, model, suite=args.config)
ik.joints_source = adapter._joints
providers = {}
if not args.no_providers:
    from cap_harness.providers.graspnet.client import ContactGraspNetProvider
    from cap_harness.providers.sam3.client import Sam3Provider
    providers = {"segmentation_provider": Sam3Provider(), "grasp_providers": {"contact-graspnet": ContactGraspNetProvider()}}
out = args.out / time.strftime("%Y%m%dT%H%M%S")
out.mkdir(parents=True, exist_ok=True)
print(f"policy server on :{args.port}; waiting for `geniesim benchmark run {args.config} --infer-host=<host>:{args.port}`", flush=True)

for episode in range(args.episodes):
    obs = adapter.reset()
    ctx = obs.task_context
    print(f"episode {ctx.task_id}: {ctx.language!r}  cameras {sorted(obs.cameras)}  grippers "
          f"{ {a: round(v, 2) for a, v in obs.robot_state.gripper_positions.items()} } raw {adapter._raw['states'].get('gripper_states')}", flush=True)
    if args.debug:
        adapter.debug_frame(out / f"episode{ctx.task_id:02d}_head.jpg")
    api = CapApi(adapter, ik_providers={"pyroki": ik}, default_camera="head",
                 default_motion_strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation", pose_planner="composed"),
                 **providers)
    t0 = time.time()
    result = run_program_with_api(api=api, program_path=args.program, extension_allowlist=ROBOSUITE_PUBLIC_TOOL_NAMES)
    progress = adapter.finish_episode()
    record = {"episode": ctx.task_id, "instruction": ctx.language, "program_ok": result.ok,
              "error": None if result.error is None else result.error.message, "result": result.result,
              "calls": len(result.calls), "steps": adapter._steps, "wall_s": round(time.time() - t0, 1),
              "success": progress["success"], "progress": progress["subtasks"], "stdout": result.stdout[-4000:]}
    (out / f"episode{ctx.task_id:02d}.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"episode {ctx.task_id}: program_ok={result.ok} success={progress['success']} steps={adapter._steps} "
          f"calls={len(result.calls)} {'' if result.error is None else result.error.message}", flush=True)
adapter.close()
