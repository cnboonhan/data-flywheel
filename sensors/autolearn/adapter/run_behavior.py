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

# BEHAVIOR_SPRING_DOORS=<spring_doors.yaml>: self-closing doors (assets/spring_doors), sprung after each reset with the
# task instance as the seed, so an instance always gets the same doors; each sprung hinge is logged to stderr
SPRING_DOORS = os.environ.get("BEHAVIOR_SPRING_DOORS")
if SPRING_DOORS:
    sys.path.insert(0, str(HERE.parent / "assets" / "spring_doors"))
    import spring_doors

    _spring_cfg = spring_doors.load(SPRING_DOORS)
    _reset = BehaviorAdapter.reset

    def reset_with_spring_doors(self, task_ref, seed):
        observation = _reset(self, task_ref, seed)
        for row in spring_doors.apply(self._require_env().scene, _spring_cfg, seed=seed):
            print(f"spring door: {json.dumps(row)}", file=sys.stderr)
        return observation

    BehaviorAdapter.reset = reset_with_spring_doors
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
    # Open house_single_floor's front door from the entryway: a latching, self-closing door (assets/spring_doors with
    # BEHAVIOR_SPRING_DOORS=.../front_door.yaml) whose lever has to be pressed before the door can be pulled open. The
    # activity is the cup's only for its task instances (the kitchen loads with them); the robot is moved to the door
    # (DOOR_TASKS) and the door witness replaces the pickup witness; target_scope only satisfies its constructor.
    BehaviorTaskMetadata(
        task_name="opening_front_door",
        activity_name="set_up_a_coffee_station_in_your_kitchen",
        language="open the door",
        target_scope="coffee_cup.n.01_1",
        target_prompt="door handle",
        support_prompt="door",
        scene_model="house_single_floor",
        load_room_types=("kitchen", "entryway"),
    ),
)
_registry_init = BehaviorTaskRegistry.__init__
BehaviorTaskRegistry.__init__ = lambda self, tasks=BEHAVIOR_TASKS + EXTRA_TASKS: _registry_init(self, tasks)
cap_run.CapApi = PyRokiCapApi
BehaviorAdapter.planner = lambda self: NoBasePlan()

# Door tasks: the door object, how far it has to open, and where the robot starts. The start is on the side the door
# swings to, beside its sweep on the latch side, facing it: `standoff` m out from the closed leaf, `lateral` m past the
# leaf's free edge (away from the hinge), heading at the door +- `yaw_deg`; each drawn per task instance, so the
# instances differ in start pose (and in the closer's spring, drawn by spring_doors). Standoff 0.65-0.85 m puts the lever
# (~0.1 m in from the free edge at 1.04 m) within the R1 Pro's right-arm reach from the base; lateral 0.05-0.25 m keeps
# the base clear of the leaf's sweep while the right shoulder (~0.25 m right of centre) stays near the lever's line;
# all untuned. open_rad 1.05 (60 degrees): wide enough to walk through.
DOOR_TASKS = {
    "opening_front_door": {"door": "door_tfdrhh_0", "open_rad": 1.05, "standoff": (0.65, 0.85), "lateral": (0.05, 0.25),
                           "yaw_deg": 8.0},
}


def door_start_pose(env, spec, instance_id):
    """The robot's start (x, y, yaw) in world for a door task instance, from the closed leaf's box and its hinge."""
    import math
    import random

    import omnigibson.utils.transform_utils as T
    import torch as th

    door = env.scene.object_registry("name", spec["door"])
    joint = next(iter(door.joints.values()))
    leaf_path = str(joint.prim.GetRelationship("physics:body1").GetTargets()[0])
    leaf = next(l for l in door.links.values() if l.prim_path == leaf_path)
    lp, lq = leaf.get_position_orientation()
    hinge = lp + T.quat2mat(lq) @ th.tensor(list(joint.prim.GetAttribute("physics:localPos1").Get()), dtype=th.float32)
    lo, hi = leaf.aabb
    centre0 = (lo + hi) / 2
    n_axis, w_axis = (0, 1) if float(hi[0] - lo[0]) < float(hi[1] - lo[1]) else (1, 0)
    # the side the door swings to: where the leaf's centre goes when it opens a little
    q0 = float(joint.get_state()[0][0])
    far = float(joint.upper_limit) if abs(float(joint.lower_limit)) <= abs(float(joint.upper_limit)) else float(joint.lower_limit)
    joint.set_pos(q0 + 0.3 * (far - q0))
    l2, h2 = leaf.aabb
    swing = 1.0 if float(((l2 + h2) / 2)[n_axis] - centre0[n_axis]) > 0 else -1.0
    joint.set_pos(q0)
    free_w = float(hi[w_axis]) if abs(float(hi[w_axis] - hinge[w_axis])) > abs(float(lo[w_axis] - hinge[w_axis])) else float(lo[w_axis])
    away = 1.0 if free_w > float(hinge[w_axis]) else -1.0          # from the hinge towards the free edge
    rng = random.Random(instance_id)
    face_n = float(lo[n_axis] if swing < 0 else hi[n_axis])
    pos = [0.0, 0.0]
    pos[n_axis] = face_n + swing * rng.uniform(*spec["standoff"])
    pos[w_axis] = free_w + away * rng.uniform(*spec["lateral"])
    heading = [0.0, 0.0]
    heading[n_axis] = -swing                                         # facing the door
    yaw = math.atan2(heading[1], heading[0]) + math.radians(rng.uniform(-spec["yaw_deg"], spec["yaw_deg"]))
    return pos[0], pos[1], yaw


_load_instance = BehaviorAdapter._load_task_instance


def load_instance_at_door(self, metadata, instance_id):
    """After the instance loads, put the robot at its door-task start (the scene's initial state is rewritten, so the
    reset that follows, and the odom frame it fixes, start there)."""
    _load_instance(self, metadata, instance_id)
    spec = DOOR_TASKS.get(metadata.task_name)
    if spec is None:
        return
    import math

    env = self._require_env()
    x, y, yaw = door_start_pose(env, spec, instance_id)
    robot = self.robot
    pos, _ = robot.get_position_orientation()
    import torch as th

    robot.set_position_orientation(th.tensor([x, y, float(pos[2])]),
                                   th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]))
    for _ in range(10):
        self._og.sim.step_physics()
    env.scene.update_initial_file()
    env.scene.reset()
    print(f"door task start: x {x:.3f} y {y:.3f} yaw {math.degrees(yaw):.1f} deg", file=sys.stderr)


BehaviorAdapter._load_task_instance = load_instance_at_door


class DoorWitness:
    """Success once the door is open past `open_rad` after its latch was released by the lever (spring_doors.state),
    held for `debounce` ticks; replaces the pickup witness for door tasks."""

    def __init__(self, task_name, door, open_rad, debounce=5):
        self.task_name, self.door, self.open_rad, self.debounce = task_name, door, open_rad, debounce
        self.ticks, self.streak, self.success_step = 0, 0, None
        self.max_open, self.min_lever, self.first_release = 0.0, 0.0, None

    @property
    def success(self):
        return self.success_step is not None

    def after_step(self, action, result):
        self.ticks += 1
        st = spring_doors.state(self.door) if SPRING_DOORS else None
        if st is None:
            return
        self.max_open = max(self.max_open, st["open_rad"])
        if st["lever_deg"] is not None:
            self.min_lever = min(self.min_lever, st["lever_deg"])
        if st["released"] and self.first_release is None:
            self.first_release = self.ticks
        if self.success:
            return
        self.streak = self.streak + 1 if (st["released"] and st["open_rad"] >= self.open_rad) else 0
        if self.streak > self.debounce:
            self.success_step = self.ticks

    def evidence(self, *, final_native_success=None):
        return {"schema_version": 1, "task_name": self.task_name, "door": self.door,
                "rule": {"open_rad": self.open_rad, "debounce_steps": self.debounce, "latch": "released by the lever"},
                "checks": {"released": self.first_release is not None, "opened": self.max_open >= self.open_rad,
                           "opened_after_release": self.success},
                "first_release_step": self.first_release, "success_step": self.success_step,
                "max_open_rad": round(self.max_open, 4), "min_lever_deg": round(self.min_lever, 2),
                "steps_observed": self.ticks, "final_native_success": final_native_success, "protocol_success": self.success}


_bind_any = BehaviorAdapter.bind_protocol_evaluator


def bind_door_witness(self, evaluator):
    spec = DOOR_TASKS.get(getattr(self._metadata, "task_name", None))
    if spec is not None:
        evaluator = DoorWitness(self._metadata.task_name, spec["door"], spec["open_rad"])
    return _bind_any(self, evaluator)


BehaviorAdapter.bind_protocol_evaluator = bind_door_witness


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
if __name__ == "__main__":      # eval_galaxea_behavior.py imports the patches above without running cap-harness
    sys.exit(main())
