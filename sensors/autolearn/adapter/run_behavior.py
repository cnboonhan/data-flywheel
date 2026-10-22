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

# BEHAVIOR_THIRD_PERSON=follow|door: a third-person camera in every observation, recorded to media/videos/third_person.mp4
# for watching only (third_person.py; behavior_to_lerobot.py keeps it out of the features and copies it to visuals/)
import third_person   # noqa: E402  (HERE is on sys.path)

if third_person.enabled():
    third_person.install(BehaviorAdapter)

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
    # The same task on the door room (assets/door_room): our own floor, walls and latching door asset (door_flylat_0,
    # assets/latch_door), the lever on the asset's own joint; BEHAVIOR_SPRING_DOORS=.../door_room.yaml. The activity
    # opening_door is registered at runtime and its one pre-sampled instance loaded as the scene file (DOOR_ROOM); the
    # seed only draws the start pose and the closer, so any seed is an instance.
    BehaviorTaskMetadata(
        task_name="opening_door",
        activity_name="opening_door",
        language="open the door",
        target_scope="door.n.01_1",
        target_prompt="door handle",
        support_prompt="door",
        scene_model="door_room",
        load_room_types=("corridor", "entryway"),
    ),
    # the same scene, instance and start pose, with the robot driving on through the doorway it opened
    BehaviorTaskMetadata(
        task_name="going_through_door",
        activity_name="opening_door",
        language="open the door and go through it",
        target_scope="door.n.01_1",
        target_prompt="door handle",
        support_prompt="door",
        scene_model="door_room",
        load_room_types=("corridor", "entryway"),
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
# `joint`: the hinge (the asset door also has the lever joint); `z`: the base height to spawn at (the door room's
# instance parks the robot off-stage, so the height can't be read from it; the R1 Pro's base sits on the floor at 0).
DOOR_TASKS = {
    "opening_front_door": {"door": "door_tfdrhh_0", "open_rad": 1.05, "standoff": (0.65, 0.85), "lateral": (0.05, 0.25),
                           "yaw_deg": 8.0},
    # door_room's geometry is known (assets/latch_door/make_latch_door.py, the door at the origin of assets/door_room):
    # `frame` gives the closed leaf's swing-side face at its free edge, the direction the leaf swings to (into the
    # corridor, -x) and the direction along the leaf from the hinge to the free edge (+y), instead of reading the
    # leaf's box, which the asset's link frame (on the hinge) and roses mislead
    "opening_door": {"door": "door_flylat_0", "joint": "j_leaf", "z": 0.0, "open_rad": 1.05, "standoff": (0.65, 0.85),
                     "lateral": (0.05, 0.25), "yaw_deg": 8.0,
                     "frame": {"face": (-0.02, 0.446), "swing": (-1.0, 0.0), "along": (0.0, 1.0)}},
    # going through: success once the base is `through_m` past the door's plane on the entryway side (the frame's face
    # x, against the swing) after the latch was released by the lever; the door may close behind. 0.6 m: the base
    # (~0.6 m long) is wholly past the leaf's sweep; untuned
    "going_through_door": {"door": "door_flylat_0", "joint": "j_leaf", "z": 0.0, "open_rad": 1.05, "through_m": 0.6,
                           "standoff": (0.65, 0.85), "lateral": (0.05, 0.25), "yaw_deg": 8.0,
                           "frame": {"face": (-0.02, 0.446), "swing": (-1.0, 0.0), "along": (0.0, 1.0)}},
}
DOOR_ROOM = "door_room"      # the scene whose activity and instance come from assets/door_room, not BEHAVIOR's data


def door_start_pose(env, spec, instance_id):
    """The robot's start (x, y, yaw) in world for a door task instance, from the closed leaf's box and its hinge."""
    import math
    import random

    import omnigibson.utils.transform_utils as T
    import torch as th

    rng = random.Random(instance_id)
    fr = spec.get("frame")
    if fr is not None:
        standoff, lateral = rng.uniform(*spec["standoff"]), rng.uniform(*spec["lateral"])
        x = fr["face"][0] + fr["swing"][0] * standoff + fr["along"][0] * lateral
        y = fr["face"][1] + fr["swing"][1] * standoff + fr["along"][1] * lateral
        yaw = math.atan2(-fr["swing"][1], -fr["swing"][0]) + math.radians(rng.uniform(-spec["yaw_deg"], spec["yaw_deg"]))
        return x, y, yaw
    door = env.scene.object_registry("name", spec["door"])
    joint = door.joints.get(spec.get("joint")) or next(iter(door.joints.values()))
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
    if metadata.scene_model != DOOR_ROOM:    # the door room's instance is the scene file itself
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

    start_pos = th.tensor([x, y, float(spec.get("z", float(pos[2])))])
    start_quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    robot.set_position_orientation(start_pos, start_quat)
    if third_person.mode() == "door" and spec.get("frame"):      # a fixed camera beside this doorway (third_person.py)
        third_person.set_fixed(*third_person.door_camera(spec["frame"]))
    # BEHAVIOR's task puts the robot back at the presampled pose on every reset: that pose becomes the door start too
    env.scene.write_task_metadata(key="robot_poses", data={"robot": [{"position": start_pos, "orientation": start_quat}]})
    for _ in range(10):
        self._og.sim.step_physics()
    env.scene.update_initial_file()
    env.scene.reset()
    print(f"door task start: x {x:.3f} y {y:.3f} yaw {math.degrees(yaw):.1f} deg", file=sys.stderr)


BehaviorAdapter._load_task_instance = load_instance_at_door
_create_native = BehaviorAdapter._create_native_environment


def create_environment_with_door_room(self, config):
    """For the door room: register its activity (OmniGibson checks the activity against the knowledge base when the
    task is built), point the scene at our dataset and at the instance file, and let the robot keep the pose
    load_instance_at_door gives it (no presampled pose)."""
    if config["scene"]["scene_model"] == DOOR_ROOM:
        self._configure_process()         # the env vars omnigibson reads on import, as the original sets them
        sys.path.insert(0, str(HERE.parent / "assets" / "door_room"))
        import door_room_task

        door_room_task.register()
        config["scene"].update(door_room_task.SCENE)
        config["scene"]["scene_file"] = str(door_room_task.INSTANCE)
        config["task"].update(door_room_task.TASK)
    return _create_native(self, config)


BehaviorAdapter._create_native_environment = create_environment_with_door_room


class DoorWitness:
    """Success once the door is open past `open_rad` after its latch was released by the lever (spring_doors.state),
    held for `debounce` ticks; replaces the pickup witness for door tasks."""

    # Upright: the robot's root tilt (roll or pitch) must stay under UPRIGHT_DEG and its height within DZ_OK of the start
    # at the success tick and through the debounce; a tilt past FALL_DEG or a drop past DZ_FALL is a fall, which fails
    # the episode at once (a v01 run drove with an arm locked on the lever, tipped over and was scored a success).
    # Contact: the leaf (a segment from the hinge, half-thickness HALF_T) may never come within MARGIN of the base's
    # footprint (its root link's box); a hit fails the episode. 10/30 degrees, 5/15 cm, 3 cm margin: untuned.
    UPRIGHT_DEG, FALL_DEG, DZ_OK, DZ_FALL, HALF_T, MARGIN = 10.0, 30.0, 0.05, 0.15, 0.02, 0.03
    LEVER_R, OFF, LEVER_Z, HAND_TOL, HANDOVER_TICKS, GRIP_ON = 0.77, 0.08, 1.0, 0.08, 5, (0.008, 0.09)
    # GRIP_ON: the finger gap (both finger joints summed, m) of a gripper closed on something: 0.1 open, ~0.0 on nothing,
    # ~0.02 on the 2 cm bar, ~0.04 on the 4 cm leaf; the sum, because a held bar sits off-centre (fingers 0.009 and
    # 0.042 on v04's lever) and a per-finger range read that as empty

    def hands(self, hinge, u, n, pos, yaw):
        """Per hand: distance to the far bar, to the near bar, to the leaf, and whether the gripper is closed on
        something (its finger gap inside GRIP_ON)."""
        robot = self.adapter.robot
        q = np.asarray(robot.get_joint_positions(), dtype=float)
        lay = self.adapter.layout
        far = np.r_[hinge + self.LEVER_R * u + self.OFF * n, self.LEVER_Z]
        near = np.r_[hinge + self.LEVER_R * u - self.OFF * n, self.LEVER_Z]
        out = {}
        for arm, og_arm in (("primary", "left"), ("secondary", "right")):
            try:
                e = np.asarray(robot.get_eef_position(og_arm), dtype=float)
            except Exception:
                e = np.asarray(robot.get_eef_position(robot.arm_names[0 if arm == "primary" else -1]), dtype=float)
            g = q[lay.grippers[og_arm]]      # the layout keys grippers by side ("left"/"right")
            on = bool(self.GRIP_ON[0] < float(np.sum(g)) < self.GRIP_ON[1])
            # distance to the leaf's segment (in the plane), at hand height
            w = e[:2] - hinge
            d = float(np.clip(np.dot(w, u), 0.0, 0.9))
            leaf = float(np.hypot(*(w - d * u)))
            out[arm] = {"far": float(np.linalg.norm(e - far)), "near": float(np.linalg.norm(e - near)), "leaf": leaf,
                        "on": on, "z": float(e[2])}
        return out

    def __init__(self, task_name, door, open_rad, debounce=5, trace=None, adapter=None, through=None, frame=None):
        self.task_name, self.door, self.open_rad, self.debounce = task_name, door, open_rad, debounce
        self.ticks, self.streak, self.success_step = 0, 0, None
        self.max_open, self.min_lever, self.first_release = 0.0, 0.0, None
        self.trace, self.adapter, self.last_open = trace, adapter, 0.0   # per-tick door trace (episode/door.jsonl)
        self.frame, self.z0, self.half = frame, None, None     # the door's 2-D frame in world (DOOR_TASKS), start height, base half-extents
        self.box_off = (0.0, 0.0)          # the footprint box's centre from the robot's position, in the base frame
        self.max_tilt, self.fell, self.hit, self.hit_step, self.fell_step = 0.0, False, False, None, None
        # the hand-over, for going_through_door: `braced` once the left gripper has closed on the leaf (its edge or the
        # near lever), `handover` once the right gripper has been closed on the FAR lever's bar for HANDOVER_TICKS
        # ticks after that; both from the eef positions against the door's geometry (assets/latch_door): levers
        # LEVER_R 0.77 m from the hinge, OFF 0.08 m off the leaf's plane, LEVER_Z 1.0 m up, within HAND_TOL 0.08 m
        # (the end-effector point sits 3-6 cm off a bar it holds: v06 traces)
        self.braced_step, self.handover_step, self.far_streak, self.hand_err = None, None, 0, None
        # through: (plane_x, swing_x, metres) for going_through_door: the base's distance past the door's plane in world,
        # against the swing (swing_x -1: the entryway is +x); None for opening_door
        self.through, self.max_through = through, None
        self.hands_now = None           # body(): per-hand distances to the levers and leaf, this tick
        self.base_geo = None            # body(): the base against the hinge and the leaf's line, this tick
        # the program's current stage: the last `stage: <name>` line it printed (it runs in this process, its stdout
        # captured in a StringIO), logged per tick in door.jsonl
        self.stage, self.stdout_pos = None, 0
        self.cross_step, self.far_at_cross = None, None     # the tick the base crossed the door plane; right hand to far bar then

    def body(self):
        """Tilt (deg), height, planar pose and whether the leaf touches the base footprint, this tick."""
        import omnigibson.utils.transform_utils as T

        robot = self.adapter.robot
        pos, quat = robot.get_position_orientation()
        roll, pitch, yaw = (float(v) for v in T.quat2euler(quat))
        tilt = max(abs(roll), abs(pitch)) * 180.0 / np.pi
        z = float(pos[2])
        if self.z0 is None:
            self.z0 = z
            # the base's footprint from base_link's world box at reset (the articulation root sits at the world origin,
            # the base pose being virtual joints, so its box is empty): the box's half extents and centre in the
            # base's frame, undoing the start yaw (+-8 degrees) the axis-aligned box is inflated by
            lo, hi = (np.asarray(v, dtype=float) for v in robot.links["base_link"].aabb)
            hw = (hi[:2] - lo[:2]) / 2
            c, s_ = abs(np.cos(yaw)), abs(np.sin(yaw))
            half = np.linalg.solve(np.array([[c, s_], [s_, c]]), hw)
            ctr = (lo[:2] + hi[:2]) / 2 - np.asarray([float(pos[0]), float(pos[1])])
            self.half = (float(half[0]), float(half[1]))
            self.box_off = (np.cos(yaw) * ctr[0] + np.sin(yaw) * ctr[1], -np.sin(yaw) * ctr[0] + np.cos(yaw) * ctr[1])
        hit = False
        self.hand_err = None
        if self.frame is not None and self.half is not None:
            st = spring_doors.state(self.door) if SPRING_DOORS else None
            if st is not None:
                f, sw, a = (np.asarray(self.frame[k], dtype=float) for k in ("face", "swing", "along"))
                hinge = f - sw * self.HALF_T - a * 0.9
                th = st["open_rad"]
                u = a * np.cos(th) + sw * np.sin(th)
                n = -sw * np.cos(th) + a * np.sin(th)        # the far face's outward normal, turning with the leaf
                try:
                    self.hands_now = self.hands(hinge, u, n, pos, yaw)
                except Exception as exc:
                    self.hands_now, self.hand_err = None, str(exc)[:80]
                # where the base is against the hinge: its centre's distance, the footprint's nearest point (outside
                # the leaf's 0.9 m sweep or not) and the side of the leaf's line it is on ("far" = the far face's side,
                # towards the latch jamb once the door is open; "near" = between the leaf and the hinge-side wall)
                cy, sy = np.cos(yaw), np.sin(yaw)
                ctr = np.asarray([float(pos[0]), float(pos[1])]) + [cy * self.box_off[0] - sy * self.box_off[1], sy * self.box_off[0] + cy * self.box_off[1]]
                near_pt = min(np.hypot(*(ctr + [cy * x - sy * y, sy * x + cy * y] - hinge))
                              for x in np.linspace(-self.half[0], self.half[0], 9) for y in np.linspace(-self.half[1], self.half[1], 9))
                self.base_geo = {"base_hinge_m": round(float(np.hypot(*(ctr - hinge))), 3),
                                 "footprint_hinge_min_m": round(float(near_pt), 3),
                                 "base_side": "far" if float(np.dot(ctr - hinge, n)) >= 0 else "near"}
                c, s_ = np.cos(-yaw), np.sin(-yaw)
                for d in np.linspace(0.0, 0.9, 19):           # points along the leaf, into the base's frame
                    w = hinge + d * u - np.asarray([float(pos[0]), float(pos[1])])
                    bx, by = c * w[0] - s_ * w[1] - self.box_off[0], s_ * w[0] + c * w[1] - self.box_off[1]
                    gap = max(abs(bx) - self.half[0], 0.0), max(abs(by) - self.half[1], 0.0)
                    if np.hypot(*gap) < self.HALF_T + self.MARGIN:
                        hit = True
                        break
        return tilt, z, hit

    def read_stage(self):
        out = sys.stdout
        try:
            pos = out.tell()
            if pos == self.stdout_pos or not hasattr(out, "getvalue"):
                return
            new = out.getvalue()[self.stdout_pos:pos]
            self.stdout_pos = pos
        except Exception:      # a real terminal or a closed buffer: no stage
            return
        for line in new.splitlines():
            if line.startswith("stage:"):
                self.stage = line[6:].strip()[:60]

    def base_through(self):
        """How far the base's centre is past the door plane on the far side (m), or None without a robot."""
        if self.through is None or self.adapter is None or self.adapter._env is None:
            return None
        plane_x, swing_x, _ = self.through
        bx = float(self.adapter.robot.get_position_orientation()[0][0])
        return (bx - plane_x) * (-swing_x)

    @property
    def success(self):
        return self.success_step is not None and not self.fell and not self.hit

    def after_step(self, action, result):
        self.ticks += 1
        st = spring_doors.state(self.door) if SPRING_DOORS else None
        if st is None:
            return
        self.max_open = max(self.max_open, st["open_rad"])
        through = self.base_through()
        tilt, z, hit = 0.0, None, False
        if self.adapter is not None and self.adapter._env is not None:
            try:
                tilt, z, hit = self.body()
            except Exception as exc:   # the checks must never break a run
                print(f"door witness body check skipped: {exc}", file=sys.stderr)
        self.max_tilt = max(self.max_tilt, tilt)
        if hit and not self.hit:
            self.hit, self.hit_step = True, self.ticks
        if (tilt > self.FALL_DEG or (z is not None and self.z0 is not None and z < self.z0 - self.DZ_FALL)) and not self.fell:
            self.fell, self.fell_step = True, self.ticks
        upright = tilt < self.UPRIGHT_DEG and (z is None or self.z0 is None or abs(z - self.z0) < self.DZ_OK)
        hn = self.hands_now
        self.read_stage()
        if hn is not None:
            left, right = hn["primary"], hn["secondary"]
            # braced: the left holds the door ALONE (closed on the leaf, its edge or the near lever, the right hand off
            # the door), the evidence that the hands switched
            right_free = not right["on"] or min(right["far"], right["near"], right["leaf"]) > self.HAND_TOL + self.HALF_T
            if self.braced_step is None and left["on"] and right_free and abs(left["z"] - self.LEVER_Z) < 0.4 \
                    and (left["leaf"] < self.HAND_TOL + self.HALF_T or left["near"] < self.HAND_TOL):
                self.braced_step = self.ticks
            self.far_streak = self.far_streak + 1 if (right["on"] and right["far"] < self.HAND_TOL) else 0
            # the hand-over counts only before the base's centre crosses the door plane (the robot goes through holding
            # the door, not re-grips it from the far side)
            if self.handover_step is None and self.braced_step is not None and self.far_streak >= self.HANDOVER_TICKS \
                    and (through is None or through <= 0.0):
                self.handover_step = self.ticks
        if through is not None:
            self.max_through = max(self.max_through, through) if self.max_through is not None else through
            # the crossing: the first tick the base's centre is past the door plane; the right hand must be at the far
            # lever then (trailing on it, its grip may be loosened to slide), within HAND_TOL + 2 cm of the bar
            if self.cross_step is None and through > 0.0:
                self.cross_step = self.ticks
                self.far_at_cross = None if hn is None else round(hn["secondary"]["far"], 3)
        if self.trace is not None:
            try:
                g = {}
                if self.adapter is not None and self.adapter._env is not None:
                    q = np.asarray(self.adapter.robot.get_joint_positions(), dtype=float)
                    lay = self.adapter.layout
                    g = {arm: q[idx].tolist() for arm, idx in lay.grippers.items()}
                with open(self.trace, "a") as f:
                    f.write(json.dumps({"tick": self.ticks, "open_rad": round(st["open_rad"], 4),
                                        "open_rate": round(st["open_rad"] - self.last_open, 5), "lever_deg": st["lever_deg"],
                                        "latched": st["latched"], "released": st["released"], "grippers": g,
                                        "through_m": None if through is None else round(through, 3),
                                        "stage": self.stage, "tilt_deg": round(tilt, 2), "base_z": None if z is None else round(z, 3),
                                        "door_hit_base": hit, "base_geo": self.base_geo,
                                        "hands": None if hn is None else {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in hn.items()},
                                        "hand_err": self.hand_err}) + "\n")
            except Exception as exc:   # tracing must never break a run
                print(f"door trace skipped: {exc}", file=sys.stderr)
        self.last_open = st["open_rad"]
        if st["lever_deg"] is not None:
            self.min_lever = min(self.min_lever, st["lever_deg"])
        if st["released"] and self.first_release is None:
            self.first_release = self.ticks
        if self.success or self.fell or self.hit:
            return
        # the latch's release is historical (it proves the door was not yanked open); the lever's and latch's final
        # state play no part: by the end the lever is let go and the latch may have caught again
        released = self.first_release is not None
        if self.through is not None:
            good = released and through is not None and through >= self.through[2] and self.handover_step is not None \
                and self.far_at_cross is not None and self.far_at_cross < self.HAND_TOL + 0.02
        else:
            good = released and st["open_rad"] >= self.open_rad
        self.streak = self.streak + 1 if (good and upright) else 0
        if self.streak > self.debounce:
            self.success_step = self.ticks

    def evidence(self, *, final_native_success=None):
        rule = {"open_rad": self.open_rad, "debounce_steps": self.debounce, "latch": "released by the lever"}
        checks = {"released": self.first_release is not None, "opened": self.max_open >= self.open_rad,
                  "upright": not self.fell, "clear_of_base": not self.hit}
        if self.through is not None:
            rule["through_m"] = self.through[2]
            checks["through"] = (self.max_through or 0.0) >= self.through[2]
            checks["braced"] = self.braced_step is not None
            checks["handover"] = self.handover_step is not None
            checks["far_lever_at_crossing"] = self.far_at_cross is not None and self.far_at_cross < self.HAND_TOL + 0.02
            checks["through_after_release"] = self.success
        else:
            checks["opened_after_release"] = self.success
        rule.update({"upright_deg": self.UPRIGHT_DEG, "fall_deg": self.FALL_DEG, "base_margin_m": self.MARGIN})
        return {"schema_version": 1, "task_name": self.task_name, "door": self.door, "rule": rule, "checks": checks,
                "fell": self.fell, "fell_step": self.fell_step, "door_hit_base": self.hit, "door_hit_base_step": self.hit_step,
                "braced_step": self.braced_step, "handover_step": self.handover_step,
                "cross_step": self.cross_step, "right_far_lever_dist_at_cross_m": self.far_at_cross,
                "max_tilt_deg": round(self.max_tilt, 2), "base_half_m": None if self.half is None else [round(v, 3) for v in self.half],
                "base_box_offset_m": [round(float(v), 3) for v in self.box_off],
                "max_through_m": None if self.max_through is None else round(self.max_through, 3),
                "first_release_step": self.first_release, "success_step": self.success_step,
                "max_open_rad": round(self.max_open, 4), "min_lever_deg": round(self.min_lever, 2),
                "steps_observed": self.ticks, "final_native_success": final_native_success, "protocol_success": self.success}


_bind_any = BehaviorAdapter.bind_protocol_evaluator


def bind_door_witness(self, evaluator):
    spec = DOOR_TASKS.get(getattr(self._metadata, "task_name", None))
    if spec is not None:
        root = getattr(self._run_observer, "root", None)
        trace = Path(root) / "episode" / "door.jsonl" if root is not None else None
        if trace is not None:
            trace.parent.mkdir(parents=True, exist_ok=True)
        through = None
        if spec.get("through_m") and spec.get("frame"):
            through = (float(spec["frame"]["face"][0]), float(spec["frame"]["swing"][0]), float(spec["through_m"]))
        door = DoorWitness(self._metadata.task_name, spec["door"], spec["open_rad"], trace=trace, adapter=self, through=through,
                           frame=spec.get("frame"))
        # cap_harness.run writes protocol.json (and protocol_success) from its own reference to the pickup witness
        evaluator.evidence = door.evidence
        evaluator = door
    return _bind_any(self, evaluator)


BehaviorAdapter.bind_protocol_evaluator = bind_door_witness
_check_native = BehaviorAdapter.check_success


def check_success_door(self):
    """Door tasks: outcome.json's task_success is the door witness's verdict (BEHAVIOR's own (open ?door) goal is met
    at ~0.1 rad, long before the task's); other tasks keep the native check (plus the grace proxy)."""
    ev = self._protocol_evaluator
    inner = getattr(ev, "inner", ev)
    if isinstance(inner, DoorWitness):
        return bool(inner.success)      # the witness itself, not the grace proxy: a run that ends within the grace counts
    return _check_native(self)


BehaviorAdapter.check_success = check_success_door


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
