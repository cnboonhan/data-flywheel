"""Genie Sim's benchmark as an ASPIRE environment.

The benchmark (`geniesim benchmark run <config> --infer-host=<host>:<port>`) drives the loop: every chunk it sends an
observation (JPEG cameras, uint16 depth, joints, the instruction, subtask scores) over a WebSocket and waits for a
chunk of absolute joint targets. `GenieSimAdapter` is that policy server turned into ASPIRE's EnvironmentAdapter:
`step()` answers the pending request with one joint target and returns the next observation; `execute_trajectory()`
sends the whole trajectory as one chunk; `reset()` takes an episode's first observation. Camera and gripper poses come from
FK on the cuRobo G2 URDF shipped with the assets (the benchmark's own robot_cfg URDF has a different torso chain and
does not match the simulated robot) plus the camera prims' offsets in the robot USD; IK is PyRoki on the same URDF.
Arms: "primary" = left, "secondary" = right. Poses are in `base_link`. Embodiment label: ASPIRE's "robosuite"
(a closed set; its 7-joint arms match the G2's).
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import cv2
import msgpack
import numpy as np
import websockets.sync.server
import yourdfpy

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    ExecutionStatus,
    IKResult,
    Observation,
    Pose,
    RobotAction,
    RobotPlanningContext,
    RobotState,
    StepResult,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import matrix_to_quaternion_wxyz, quaternion_wxyz_to_matrix

EMBODIMENT = "robosuite"          # ASPIRE's embodiment labels are a closed set; this one is 7 joints per arm
BASE_FRAME = "base_link"
ARMS = ("primary", "secondary")   # left, right
ARM_JOINTS = {
    "primary": tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)),
    "secondary": tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)),
}
WAIST_JOINTS = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))
GRIPPER_JOINTS = {"primary": "idx41_gripper_l_outer_joint1", "secondary": "idx81_gripper_r_outer_joint1"}
EE_LINKS = {"primary": "gripper_l_center_link", "secondary": "gripper_r_center_link"}
GRIPPER_OPEN = -0.785             # the benchmark's gripper value when open (it reports the omnipicker at -0.78 open, 0 closed)
GRIPPER_STEPS = 12                # ticks a gripper command is held for, so the fingers finish moving (episodes are ~330 steps)
CONTROL_FREQUENCY_HZ = 30.0       # one chunk entry per benchmark step, replayed at its 30 fps render rate
# Payload camera -> (public name, USD camera prim, URDF link the prim hangs from); from robot_cfg/G2_omnipicker.json.
CAMERAS = {
    "head": ("head", "head_front_Camera", "head_link3"),
    "hand_left": ("left_wrist", "Left_Camera", "gripper_l_base_link"),
    "hand_right": ("right_wrist", "Right_Camera", "gripper_r_base_link"),
}
OPENGL_TO_IMAGE = np.diag([1.0, -1.0, -1.0, 1.0])   # USD cameras look down -Z with +Y up; images: x right, y down, z forward


def _np_hook(obj):
    """msgpack-numpy encodes arrays as dicts; the benchmark's payload is mostly bytes and lists, but be safe."""
    for key in ("nd", b"nd"):
        if key in obj:
            dtype = np.dtype(obj.get("type", obj.get(b"type")))
            return np.frombuffer(obj.get("data", obj.get(b"data")), dtype=dtype).reshape(obj.get("shape", obj.get(b"shape")))
    return obj


def _transform(translate, quat_wxyz) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quaternion_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=float))
    m[:3, 3] = translate
    return m


def _pose(matrix: np.ndarray) -> Pose:
    return Pose(position=matrix[:3, 3].copy(), quaternion_wxyz=matrix_to_quaternion_wxyz(matrix[:3, :3]), frame=BASE_FRAME)


def camera_prims(usda: Path) -> dict[str, dict]:
    """focalLength, apertures (mm) and the prim's transform in its parent link, per camera prim in the robot USD."""
    text = usda.read_text()
    out = {}
    for name in {c[1] for c in CAMERAS.values()}:
        i = text.find(f'def Camera "{name}"')
        if i < 0:
            raise ValueError(f"camera prim {name!r} not in {usda}")
        block = text[i:i + 4000]
        f = lambda key: float(re.search(rf"float {key} = ([-\d.e]+)", block).group(1))
        tr = re.search(r"xformOp:translate = \(([^)]*)\)", block).group(1)
        q = re.search(r"xformOp:orient = \(([^)]*)\)", block).group(1)
        out[name] = {"focal_mm": f("focalLength"), "h_aperture_mm": f("horizontalAperture"),
                     "v_aperture_mm": f("verticalAperture"),
                     "link_from_camera": _transform([float(v) for v in tr.split(",")], [float(v) for v in q.split(",")])}
    return out


class G2Model:
    """The G2's kinematics (yourdfpy on the benchmark's URDF): camera poses, joint limits, a PyRoki robot for IK."""

    def __init__(self, urdf: Path, usda: Path):
        self.urdf = yourdfpy.URDF.load(str(urdf), load_meshes=False, build_scene_graph=True)
        self.cameras = camera_prims(usda)
        self.limits = {}
        for arm, names in ARM_JOINTS.items():
            lo = np.array([self.urdf.joint_map[n].limit.lower for n in names])
            hi = np.array([self.urdf.joint_map[n].limit.upper for n in names])
            self.limits[arm] = (lo, hi)
        self._lock = threading.Lock()

    def fk(self, joints: Mapping[str, float], link: str) -> np.ndarray:
        with self._lock:
            cfg = {n: 0.0 for n in self.urdf.actuated_joint_names}
            cfg.update({n: float(v) for n, v in joints.items() if n in cfg})
            self.urdf.update_cfg(cfg)
            return np.asarray(self.urdf.get_transform(link, BASE_FRAME))

    def camera(self, prim: str, link: str, joints: Mapping[str, float], width: int, height: int):
        p = self.cameras[prim]
        fx = width * p["focal_mm"] / p["h_aperture_mm"]
        fy = height * p["focal_mm"] / p["v_aperture_mm"]
        intrinsics = np.array([[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]])
        base_from_camera = self.fk(joints, link) @ p["link_from_camera"] @ OPENGL_TO_IMAGE
        return intrinsics, base_from_camera


class G2IkProvider:
    """ASPIRE IKProvider on PyRoki with the G2 URDF: solves one arm's 7 joints to a gripper-centre pose in base_link."""

    def __init__(self, model: G2Model):
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk

        self._jnp = jnp
        self.robot = pk.Robot.from_urdf(model.urdf)
        self.names = list(self.robot.joints.actuated_names)
        self.link_index = {arm: self.robot.links.names.index(link) for arm, link in EE_LINKS.items()}
        self._lock = threading.Lock()

        # Only the commanded arm may move: every other joint (the waist is on the arm's chain too) is pinned to
        # the seed by a heavy rest cost; the benchmark executes the arm joints alone.
        self.rest_weight = {}
        for arm in ARMS:
            w = np.full(len(self.names), 100.0)
            for n in ARM_JOINTS[arm]:
                w[self.names.index(n)] = 0.001
            self.rest_weight[arm] = jnp.asarray(w)

        @jax.jit
        def solve(robot, link_index, target_wxyz, target_xyz, seed, rest_weight):
            var = robot.joint_var_cls(0)
            costs = [pk.costs.pose_cost_analytic_jac(
                         robot, var, jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_xyz),
                         link_index, pos_weight=50.0, ori_weight=10.0),
                     pk.costs.rest_cost(var, seed, rest_weight),
                     pk.costs.limit_constraint(robot, var)]
            sol = (jaxls.LeastSquaresProblem(costs=costs, variables=[var]).analyze()
                   .solve(verbose=False, linear_solver="dense_cholesky",
                          trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                          initial_vals=jaxls.VarValues.make([var.with_value(seed)])))
            return sol[var]

        self._solve = solve
        self.joints_source = None     # set to the adapter's joint reading: the waist (on each arm's chain) isn't in RobotState

    def solve_ik(self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary", **_) -> IKResult:
        if arm not in ARMS:
            return IKResult(ok=False, arm=arm, embodiment=EMBODIMENT,
                            error=ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}"))
        if target_pose.frame != BASE_FRAME:
            return IKResult(ok=False, arm=arm, embodiment=EMBODIMENT,
                            error=ApiError(ErrorCode.INVALID_REQUEST, f"target frame must be {BASE_FRAME!r}"))
        seed = np.zeros(len(self.names))
        for n, v in (self.joints_source() if self.joints_source else {}).items():
            if n in self.names:
                seed[self.names.index(n)] = v
        for a in ARMS:
            for n, v in zip(ARM_JOINTS[a], robot_state.joint_positions[a]):
                seed[self.names.index(n)] = v
        with self._lock:
            jnp = self._jnp
            sol = np.asarray(self._solve(self.robot, jnp.asarray(self.link_index[arm]),
                                         jnp.asarray(target_pose.quaternion_wxyz), jnp.asarray(target_pose.position),
                                         jnp.asarray(seed), self.rest_weight[arm]))
            pose = np.asarray(self.robot.forward_kinematics(jnp.asarray(sol))[self.link_index[arm]])
        joints = np.array([sol[self.names.index(n)] for n in ARM_JOINTS[arm]], dtype=float)
        pos_err = float(np.linalg.norm(pose[4:] - target_pose.position))
        ori_err = float(2 * np.arccos(np.clip(abs(np.dot(pose[:4], target_pose.quaternion_wxyz)), 0, 1)))
        diag = {"position_error_m": pos_err, "orientation_error_rad": ori_err, "provider": "pyroki-g2"}
        if pos_err > 0.01 or ori_err > 0.05:
            return IKResult(ok=False, arm=arm, embodiment=EMBODIMENT, diagnostics=diag,
                            error=ApiError(ErrorCode.IK_UNREACHABLE,
                                           f"IK residual {pos_err * 100:.1f} cm, {np.degrees(ori_err):.1f} deg"))
        return IKResult(ok=True, joint_positions=joints, arm=arm, embodiment=EMBODIMENT, diagnostics=diag)


class CorobotBridge:
    """The WebSocket side: one observation in, exactly one reply out, in order."""

    def __init__(self, port: int):
        self.port = port
        self.observations: queue.Queue = queue.Queue()
        self.replies: queue.Queue = queue.Queue()
        self.connected = threading.Event()
        self._server = None

    def start(self):
        self._server = websockets.sync.server.serve(self._handle, "0.0.0.0", self.port, max_size=None)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self):
        if self._server is not None:
            self._server.shutdown()

    def _handle(self, ws):
        self.connected.set()
        for raw in ws:
            req = msgpack.unpackb(raw, raw=False, object_hook=_np_hook)
            self.observations.put(req.get("params", {}))
            reply = self.replies.get()
            ws.send(msgpack.packb({"result": reply}, use_bin_type=True))

    def next_observation(self, timeout: float):
        return self.observations.get(timeout=timeout)

    def reply(self, result: dict):
        self.replies.put(result)


class GenieSimAdapter:
    """ASPIRE EnvironmentAdapter over the benchmark's policy socket."""

    embodiment = EMBODIMENT
    control_frequency = CONTROL_FREQUENCY_HZ
    control_period_s = 1.0 / CONTROL_FREQUENCY_HZ

    def __init__(self, bridge: CorobotBridge, model: G2Model, *, suite: str, observation_timeout_s: float = 600.0):
        self.bridge = bridge
        self.model = model
        self.suite = suite
        self.timeout = observation_timeout_s
        self._raw = None            # the latest observation (the benchmark's params)
        self._images = {}           # last decoded cameras, kept when a request carries no images
        self._owed = False          # a reply is owed for self._raw
        self._terminated = False
        self._steps = 0
        self._t0 = time.time()
        self.debug_dir = None       # when set, every gripper command saves the cameras with the grippers projected
        self._snapshots = 0

    # -- episode lifecycle -------------------------------------------------------------------------------------------
    def reset(self, task_ref: object = None, seed: int = 0) -> Observation:
        """Take the next episode's first observation (the benchmark decides episodes, instances and seeds)."""
        if self._owed:
            self._hold()
        self._raw, self._owed = self.bridge.next_observation(self.timeout), True
        self._terminated, self._steps, self._t0 = bool(self._raw.get("episode_done")), 0, time.time()
        self._decode_images()
        return self.get_observation()

    def finish_episode(self) -> Mapping[str, object]:
        """Hold still until the benchmark ends the episode; return its last progress."""
        while not self._terminated:
            self._exchange(self._chunk(1))
        if self._owed:
            self._hold()
        return self.progress()

    def close(self) -> None:
        self.bridge.stop()

    # -- observations ------------------------------------------------------------------------------------------------
    def _decode_images(self):
        images = self._raw.get("images", {})
        for key, (name, prim, link) in CAMERAS.items():
            rgb, depth = images.get(key), images.get(f"{key}_depth")
            if rgb is None:
                continue
            bgr = cv2.imdecode(np.frombuffer(rgb["image_data"], np.uint8), cv2.IMREAD_COLOR)
            h, w = bgr.shape[:2]
            if depth is not None:
                d = np.frombuffer(depth["image_data"], np.uint16).reshape(depth["height"], depth["width"]).astype(np.float64) / 1000.0
                if d.shape != (h, w):
                    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                d = self._images.get(key, (None, np.zeros((h, w))))[1]
            self._images[key] = (np.ascontiguousarray(bgr[:, :, ::-1]), d)

    def _joints(self) -> dict[str, float]:
        st = self._raw["states"]
        arm, waist = st["arm_joint_states"], st.get("waist_joint_states", [])
        joints = {n: arm[i] for i, n in enumerate(ARM_JOINTS["primary"] + ARM_JOINTS["secondary"])}
        joints.update({n: waist[i] for i, n in enumerate(WAIST_JOINTS) if i < len(waist)})
        return joints

    def get_robot_state(self) -> RobotState:
        st = self._raw["states"]
        arm = np.asarray(st["arm_joint_states"], dtype=float)
        grip = np.asarray(st.get("gripper_states", [0, 0]), dtype=float)
        poses = {}
        joints = self._joints()
        # Poses from our own FK, the model IK plans on, so planned and measured poses agree. The benchmark's
        # `end_pose` (its own FK, a different tool frame) is kept for diagnostics only.
        for a in ARMS:
            poses[a] = _pose(self.model.fk(joints, EE_LINKS[a]))
        return RobotState(
            joint_positions={"primary": arm[:7], "secondary": arm[7:14]},
            joint_velocities={a: np.zeros(7) for a in ARMS},
            end_effector_poses=poses,
            gripper_positions={a: float(np.clip(grip[i] / GRIPPER_OPEN, 0.0, 1.0)) for i, a in enumerate(ARMS)},
            base_frame=BASE_FRAME,
            joint_names={a: ARM_JOINTS[a] for a in ARMS},
            timestamp_s=self.current_time_s,
            embodiment=EMBODIMENT,
        )

    def get_task_context(self) -> TaskContext:
        return TaskContext(suite=self.suite, task_id=int(self._raw.get("episode_idx", 0)),
                           task_name=str(self._raw.get("task_name", self.suite)), language=str(self._raw.get("prompt", "")),
                           family="geniesim", metadata={"robot_type": self._raw.get("robot_type"), "progress": self.progress()})

    def get_observation(self) -> Observation:
        joints = self._joints()
        cameras = {}
        for key, (name, prim, link) in CAMERAS.items():
            if key not in self._images:
                continue
            rgb, depth = self._images[key]
            intrinsics, base_from_camera = self.model.camera(prim, link, joints, rgb.shape[1], rgb.shape[0])
            cameras[name] = CameraObservation(rgb=rgb, depth_m=depth, intrinsics=intrinsics, frame=f"camera/{name}",
                                              camera_pose=_pose(base_from_camera), timestamp_s=self.current_time_s)
        return Observation(cameras=cameras, robot_state=self.get_robot_state(), task_context=self.get_task_context(),
                           timestamp_s=self.current_time_s)

    @property
    def current_time_s(self) -> float:
        return self._steps * self.control_period_s

    def progress(self) -> Mapping[str, object]:
        items = [{"name": p.get("name"), "score": p.get("score"), "status": p.get("status")}
                 for p in self._raw.get("task_progress", [])]
        return {"episode_done": bool(self._raw.get("episode_done")), "subtasks": items,
                "success": bool(items) and all(float(p["score"] or 0) >= 1 for p in items)}

    def check_success(self) -> bool:
        return bool(self.progress()["success"])

    # -- planning calibration ----------------------------------------------------------------------------------------
    def get_planning_context(self) -> RobotPlanningContext:
        return RobotPlanningContext(embodiment=EMBODIMENT, model="G2_omnipicker",
                                    joint_names={a: ARM_JOINTS[a] for a in ARMS},
                                    base_transforms={a: np.eye(4) for a in ARMS},
                                    end_effector_links=dict(EE_LINKS))

    @property
    def arms(self) -> tuple[str, ...]:
        return ARMS

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.model.limits["primary"]

    # -- control -------------------------------------------------------------------------------------------------------
    def _chunk(self, steps: int, targets: Mapping[str, np.ndarray] | None = None,
               grippers: Mapping[str, float] | None = None, rows: np.ndarray | None = None) -> dict:
        """A reply: `steps` rows of absolute joints. `rows` (steps x 7) overrides one arm's rows for a trajectory."""
        st = self._raw["states"]
        arm = np.asarray(st["arm_joint_states"], dtype=float).copy()
        grip = np.asarray(st.get("gripper_states", [0, 0]), dtype=float).copy()
        waist = list(st.get("waist_joint_states", []))
        for i, a in enumerate(ARMS):
            if targets and a in targets:
                arm[7 * i:7 * i + 7] = targets[a]
            if grippers and grippers.get(a) is not None:
                grip[i] = float(grippers[a]) * GRIPPER_OPEN
        left, right = [arm[:7].tolist()] * steps, [arm[7:].tolist()] * steps
        if rows is not None:
            if "primary" in (targets or {}):
                left = rows.tolist()
            else:
                right = rows.tolist()
        reply = {"left_arm": {"kind": "JOINT_ABS", "values": left}, "right_arm": {"kind": "JOINT_ABS", "values": right},
                 "left_effector": [[float(grip[0])]] * steps, "right_effector": [[float(grip[1])]] * steps,
                 "need_depth": True}
        if waist:
            reply["waist"] = {"kind": "JOINT_ABS", "values": [waist] * steps}
        return reply

    def _hold(self):
        self.bridge.reply(self._chunk(1))
        self._owed = False

    def _exchange(self, reply: dict) -> int:
        """Send a chunk for the owed observation, take the next one; returns the steps the chunk advanced."""
        steps = len(reply["left_arm"]["values"])
        self.bridge.reply(reply)
        self._raw, self._owed = self.bridge.next_observation(self.timeout), True
        self._steps += steps
        self._decode_images()
        if self._raw.get("episode_done"):
            self._terminated = True
        return steps

    def step(self, action: RobotAction) -> StepResult:
        if action.embodiment != EMBODIMENT:
            return StepResult(ok=False, error=ApiError(ErrorCode.INVALID_REQUEST, "wrong embodiment"))
        if self._terminated:
            return StepResult(ok=False, terminated=True, error=ApiError(ErrorCode.TERMINATED, "episode is over"))
        targets = {a: np.asarray(c.target, dtype=float) for a, c in action.arms.items()}
        grippers = {a: c.gripper_position for a, c in action.arms.items()}
        self._exchange(self._chunk(1, targets, grippers))
        return StepResult(ok=True, observation=self.get_observation(), terminated=self._terminated,
                          diagnostics={"control_period_s": self.control_period_s, "step_index": self._steps})

    def execute_trajectory(self, trajectory: Trajectory) -> ExecutionResult:
        arm = trajectory.arm
        if arm not in ARMS:
            return ExecutionResult(ok=False, steps_executed=0, status=ExecutionStatus.INVALID_REQUEST,
                                   error=ApiError(ErrorCode.INVALID_REQUEST, f"unknown arm {arm!r}"))
        state = self.get_robot_state()
        start_err = float(np.max(np.abs(trajectory.expected_start.joint_positions[arm] - state.joint_positions[arm])))
        if start_err > 0.05:
            return ExecutionResult(ok=False, steps_executed=0, status=ExecutionStatus.STALE_PLAN,
                                   error=ApiError(ErrorCode.STALE_PLAN, "trajectory start no longer matches the robot",
                                                  details={"joint_error_rad": start_err}), final_errors={arm: start_err})
        if self._terminated:
            return ExecutionResult(ok=False, steps_executed=0, status=ExecutionStatus.TERMINATED, terminated=True,
                                   error=ApiError(ErrorCode.TERMINATED, "episode is over"))
        rows = np.asarray(trajectory.joint_positions, dtype=float)
        grip = None if trajectory.gripper_positions is None else float(trajectory.gripper_positions[-1])
        done = 0
        for start in range(0, len(rows), 300):            # chunks the benchmark replays without a round trip
            part = rows[start:start + 300]
            done += self._exchange(self._chunk(len(part), {arm: part[-1]}, {arm: grip}, rows=part))
            if self._terminated:
                break
        final = self.get_robot_state()
        err = float(np.max(np.abs(rows[-1] - final.joint_positions[arm])))
        return ExecutionResult(ok=not self._terminated or done == len(rows), steps_executed=done,
                               status=ExecutionStatus.TERMINATED if self._terminated else ExecutionStatus.SUCCESS,
                               final_observation=self.get_observation(), terminated=self._terminated,
                               final_errors={arm: err})

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        bad = [a for a in positions if a not in ARMS]
        if bad or not all(np.isfinite(v) and 0 <= v <= 1 for v in positions.values()):
            return ExecutionResult(ok=False, steps_executed=0, status=ExecutionStatus.INVALID_REQUEST,
                                   error=ApiError(ErrorCode.INVALID_REQUEST, "gripper positions must be arms -> [0, 1]"))
        if self._terminated:
            return ExecutionResult(ok=False, steps_executed=0, status=ExecutionStatus.TERMINATED, terminated=True,
                                   error=ApiError(ErrorCode.TERMINATED, "episode is over"))
        if self.debug_dir is not None:
            self.debug_frame(Path(self.debug_dir) / f"episode{int(self._raw.get('episode_idx', 0)):02d}_grip{self._snapshots:02d}_{'close' if min(positions.values()) < 0.5 else 'open'}.jpg")
            self._snapshots += 1
        steps = self._exchange(self._chunk(GRIPPER_STEPS, None, dict(positions)))   # the fingers need a few ticks
        return ExecutionResult(ok=True, steps_executed=steps, status=ExecutionStatus.SUCCESS,
                               final_observation=self.get_observation(), terminated=self._terminated)

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        return self.set_grippers({arm: position})

    def get_task_metadata(self) -> Mapping[str, object]:
        """ASPIRE's `robosuite.get_task_metadata` extension: the benchmark's progress, nothing privileged."""
        return {"suite": self.suite, "embodiment": EMBODIMENT, "base_frame": BASE_FRAME, "arm_names": ARMS,
                "control_frequency_hz": self.control_frequency, **self.progress()}

    def get_controller_metadata(self) -> Mapping[str, object]:
        """What ASPIRE's interpolation planner reads: joint-limit action bounds, the control rate, the arm order."""
        lo, hi = self.model.limits["primary"]
        return {"embodiment": EMBODIMENT, "base_frame": BASE_FRAME, "arm_names": ARMS,
                "control_frequency_hz": self.control_frequency, "control_period_s": self.control_period_s,
                "action_lower_bounds": np.concatenate([lo, lo]), "action_upper_bounds": np.concatenate([hi, hi]),
                "suite": self.suite, "progress": self.progress()}

    def debug_frame(self, path: Path):
        """The head image with both gripper centres projected: a check of the camera calibration."""
        obs = self.get_observation()
        if "head" not in obs.cameras:
            return
        for name, cam in obs.cameras.items():
            img = np.ascontiguousarray(cam.rgb[:, :, ::-1])
            cam_from_base = np.linalg.inv(np.r_[np.c_[quaternion_wxyz_to_matrix(cam.camera_pose.quaternion_wxyz), cam.camera_pose.position], [[0, 0, 0, 1]]])
            for arm, pose in obs.robot_state.end_effector_poses.items():
                p = cam_from_base @ np.r_[pose.position, 1.0]
                if p[2] > 0:
                    uv = cam.intrinsics @ (p[:3] / p[2])
                    cv2.circle(img, (int(uv[0]), int(uv[1])), 8, (0, 0, 255) if arm == "primary" else (255, 0, 0), 2)
            cv2.imwrite(str(path if name == "head" else path.with_name(path.stem + f"_{name}.jpg")), img)
        cam = obs.cameras["head"]
        json.dump({"intrinsics": cam.intrinsics.tolist(), "camera_pose": cam.camera_pose.position.tolist(),
                   "ee": {a: p.position.tolist() for a, p in obs.robot_state.end_effector_poses.items()},
                   "ee_fk": {a: self.model.fk(self._joints(), EE_LINKS[a])[:3, 3].tolist() for a in ARMS},
                   "ee_fk_no_waist": {a: self.model.fk({n: v for n, v in self._joints().items() if n not in WAIST_JOINTS}, EE_LINKS[a])[:3, 3].tolist() for a in ARMS},
                   "head_link3_fk": self.model.fk(self._joints(), "head_link3")[:3, 3].tolist(),
                   "states": {k: v for k, v in self._raw["states"].items() if k != "end_pose"},
                   "end_pose": self._raw["states"].get("end_pose")},
                  open(path.with_suffix(".json"), "w"), indent=1)
