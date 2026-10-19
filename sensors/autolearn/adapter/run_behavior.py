"""`cap-harness` for BEHAVIOR-1K's R1 Pro with PyRoki IK in place of OmniGibson's cuRobo.

Same command line as `cap-harness` (`run --benchmark behavior ...`, `doctor`, ...) and the same recorded run
(ASPIRE's BehaviorAdapter, pickup witness, artifacts); only what cuRobo did changes, because OmniGibson's cuRobo
drops the R1 Pro embodiment on RTX 50-series GPUs:

- arm IK is `adapter/ik.py` (PyRoki on the R1 Pro URDF, torso pinned), the provider named `pyroki`, and the default
  `MotionStrategy` is PyRoki + ASPIRE's interpolation planner, as on Genie Sim; `curobo` is not configured, so
  `MotionStrategy(ik_solver="curobo")` is refused rather than silently remapped;
- the base never gets a cuRobo plan, so `behavior.navigate_to_pose` always follows the scene's traversability map
  in servoed hops (its own fallback; `planner="servo"` goes straight there).

Run it through `cap-harness-behavior.sh`, which sandboxes it like the LIBERO loop:

    CAP_HARNESS=../cap-harness-behavior.sh    # from aspire/, in place of .venv-behavior/bin/cap-harness
    $CAP_HARNESS run --benchmark behavior --suite turning_on_radio --task-id 0 --seed 26 --program policy.py \\
        --output-root outputs/runs/<stamp>/behavior/turning_on_radio/stage1/seed_26/attempts --flat-run-dir \\
        --max-steps 6000 --camera-width 512 --camera-height 512
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cap_harness.run as cap_run  # noqa: E402
from cap_harness.api import CapApi  # noqa: E402
from cap_harness.artifacts import TracedProvider  # noqa: E402
from cap_harness.behavior.adapter import BehaviorAdapter, _pose_matrix  # noqa: E402
from cap_harness.behavior.registry import BEHAVIOR_TASKS, BehaviorTaskMetadata, BehaviorTaskRegistry  # noqa: E402
from cap_harness.cli import main  # noqa: E402
from cap_harness.contracts import MotionStrategy, RobotPlanningContext  # noqa: E402
from cap_harness.errors import ApiError, ErrorCode  # noqa: E402
from ik import UrdfIkProvider  # noqa: E402

URDF = (Path(os.environ.get("OMNIGIBSON_DATA_PATH") or HERE.parent / "behavior-data")
        / "omnigibson-robot-assets/models/r1pro/urdf/r1pro_with_meta_links.urdf")


def pyroki_for(adapter: BehaviorAdapter) -> UrdfIkProvider:
    robot = adapter.robot

    def world_pose(link: str) -> np.ndarray:
        return _pose_matrix(*robot.links[link].get_position_orientation())

    def all_joints():     # ASPIRE's RobotState carries the arms only; the torso and base come from the robot
        return dict(zip(list(robot.joints.keys()), np.asarray(robot.get_joint_positions()).tolist()))

    def context() -> RobotPlanningContext:
        # The adapter's base transform is odom from the articulation root, which OmniGibson leaves at the world
        # origin with the base pose in virtual joints; the URDF's root is base_link, so take its live pose instead.
        ctx = adapter.get_planning_context()
        odom_from_root = adapter.odom_from_world @ world_pose("base_link")
        return RobotPlanningContext(embodiment=ctx.embodiment, model=ctx.model, joint_names=ctx.joint_names,
                                    base_transforms={arm: odom_from_root for arm in ctx.joint_names},
                                    end_effector_links=ctx.end_effector_links)

    ik = UrdfIkProvider(URDF, context_source=context, joints_source=all_joints)
    ctx = adapter.get_planning_context()
    for arm, link in ctx.end_effector_links.items():
        if link not in ik.robot.links.names:      # OmniGibson's eef meta link isn't in the URDF: the gripper link is,
            stand_in = f"{'left' if arm == 'primary' else 'right'}_gripper_link"      # at a fixed offset measured here
            ik.link_map[link] = stand_in
            ik.link_offsets[link] = np.linalg.inv(world_pose(link)) @ world_pose(stand_in)
    missing = [n for names in ctx.joint_names.values() for n in names if n not in ik.names]
    missing += [l for l in ik.link_map.values() if l not in ik.robot.links.names]
    if missing:
        sys.exit(f"{URDF} lacks {missing}: pick the URDF whose names match OmniGibson's R1 Pro")
    return ik


class PyRokiCapApi(CapApi):
    """ASPIRE's CapApi with the BEHAVIOR motion providers replaced (run.py passes cuRobo for all three)."""

    def __init__(self, adapter, **kw):
        if isinstance(adapter, BehaviorAdapter):
            recorder = kw.get("recorder")
            ik = pyroki_for(adapter)
            kw.update(ik_providers={"pyroki": TracedProvider(ik, recorder, "pyroki") if recorder else ik},
                      trajectory_planning_providers={}, integrated_pose_planning_providers={},
                      default_motion_strategy=MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation",
                                                             pose_planner="composed"))
        super().__init__(adapter, **kw)


class NoBasePlan:
    """What navigate_to_pose asks the cuRobo planner: no plan, so it takes the traversability route with servo hops."""

    def plan_base(self, x, y, yaw):
        return [], ApiError(ErrorCode.PLANNING_FAILED, "no cuRobo on this GPU; servoing", details={"status": "unavailable"})


class GraceEvaluator:
    """The pickup witness, with the episode's end held off for `grace` ticks after it first succeeds.

    ASPIRE ends a BEHAVIOR episode on the tick the witness latches (the radio 5 mm up), so a demonstration stops at the
    start of the lift. Through this proxy the adapter sees success only `grace` ticks later, so the program's lift
    plays out; the witness itself (and protocol.json, written from it) still records the first success.
    """

    def __init__(self, inner, grace: int):
        self.inner, self.grace, self.ticks, self.first = inner, grace, 0, None

    def after_step(self, *a, **kw):
        out = self.inner.after_step(*a, **kw)
        if self.first is None and bool(getattr(self.inner, "success", False)):
            self.first = self.ticks
        self.ticks += 1
        return out

    @property
    def success(self):
        return self.first is not None and self.ticks - self.first >= self.grace

    def __getattr__(self, name):
        return getattr(self.inner, name)


# CAP_HARNESS_SUCCESS_GRACE_TICKS: ticks the episode runs on after the witness first succeeds (0: end at once, as
# upstream); collect_behavior.sh sets it so demonstrations include the lift
GRACE = int(os.environ.get("CAP_HARNESS_SUCCESS_GRACE_TICKS", "0"))
if GRACE > 0:
    _bind = BehaviorAdapter.bind_protocol_evaluator
    BehaviorAdapter.bind_protocol_evaluator = lambda self, ev: _bind(self, GraceEvaluator(ev, GRACE))
    _check = BehaviorAdapter.check_success
    BehaviorAdapter.check_success = lambda self: _check(self) or bool(
        getattr(getattr(self._protocol_evaluator, "inner", None), "success", False))

# Whole-body record: ASPIRE's per-tick episode carries the arms and grippers only, while the base drives and the torso
# leans through behavior.* calls; a VLA trained without them never learns to drive or lean. Each tick appends the base
# pose in odom, the base and torso joint readings and targets to episode/whole_body.jsonl beside steps.jsonl.
_step = BehaviorAdapter.step


def step_recording_whole_body(self, action):
    result = _step(self, action)
    root = getattr(self._run_observer, "root", None)
    if root is not None and self._env is not None:
        try:
            layout = self.layout
            q = np.asarray(self.robot.get_joint_positions(), dtype=float)
            target = np.asarray(self._q_target, dtype=float)
            row = {"tick": int(self._elapsed_steps), "base_pose_odom": [float(v) for v in self.base_pose_odom()],
                   "base_q": q[layout.base].tolist(), "base_target": target[layout.base].tolist(),
                   "torso_q": q[layout.trunk].tolist(), "torso_target": target[layout.trunk].tolist()}
            with open(Path(root) / "episode" / "whole_body.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as exc:  # recording must never break a run
            print(f"whole-body record skipped: {exc}", file=sys.stderr)
    return result


BehaviorAdapter.step = step_recording_whole_body
# Pickup tasks beyond ASPIRE's two, registered here so the submodule stays untouched. Each is a BEHAVIOR-1K activity with
# task instances on disk, reduced to a pickup of one object of its scope (ASPIRE's pickup witness: held and >= 5 mm up).
EXTRA_TASKS = (
    BehaviorTaskMetadata(
        task_name="picking_up_cup",
        activity_name="set_up_a_coffee_station_in_your_kitchen",   # starts with the cup on a kitchen countertop
        language="pick up the cup",
        target_scope="coffee_cup.n.01_1",
        target_prompt="coffee cup",
        support_prompt="countertop",
        scene_model="house_single_floor",
        load_room_types=("kitchen",),
    ),
)
_registry_init = BehaviorTaskRegistry.__init__
BehaviorTaskRegistry.__init__ = lambda self, tasks=BEHAVIOR_TASKS + EXTRA_TASKS: _registry_init(self, tasks)
cap_run.CapApi = PyRokiCapApi
BehaviorAdapter.planner = lambda self: NoBasePlan()


def tolerate_unloadable_mdl():
    """On arm64 (GB300) a kitchen object's material has an MDL module neuray cannot load ('rtx::neuraylib::MdlModuleId'
    for '' is Invalid) and OmniGibson's MaterialPrim raises, so the scene never loads. Keep the material (it is created
    and bound) and skip only its shader node, which OmniGibson reads for material inputs, not for loading or physics."""
    from omnigibson.prims import material_prim as mp

    post_load = mp.MaterialPrim._post_load

    def _post_load(self):
        try:
            post_load(self)
        except Exception as exc:
            if "MdlModuleId" not in str(exc):
                raise
            print(f"material {self.prim_path}: MDL module not loadable, shader node skipped", file=sys.stderr)
            self._shader_node = None

    mp.MaterialPrim._post_load = _post_load


tolerate_unloadable_mdl()
sys.exit(main())
