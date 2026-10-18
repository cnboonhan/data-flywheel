"""An ASPIRE IKProvider on PyRoki for any URDF: the Genie Sim G2 and OmniGibson's R1 Pro use it alike.

Solves one arm's joints to an end-effector pose while every other joint (torso and waist are on each arm's chain)
is pinned to the robot's current reading by a heavy rest cost. The arm's joint names, end-effector link and base
transform come from the adapter's RobotPlanningContext at call time, the current reading of every joint from
`joints_source` (ASPIRE's RobotState only carries the arm joints).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping

import numpy as np
import yourdfpy

from cap_harness.contracts import IKResult, Pose, RobotPlanningContext, RobotState
from cap_harness.errors import ApiError, ErrorCode
from cap_harness.geometry import quaternion_wxyz_to_matrix

REST_WEIGHT = 100.0        # on pinned joints; the commanded arm's joints get 0.001 (a tiny regulariser)
# a solve further off than this is reported unreachable and nothing moves; 1.5 cm lets a grasp at the edge of the
# G2's reach (residuals of 1.1 cm seen there) go ahead and the caller's own pose tolerance judge it
POSITION_TOLERANCE_M = 0.015
ORIENTATION_TOLERANCE_RAD = 0.08


class UrdfIkProvider:
    def __init__(self, urdf: yourdfpy.URDF | str, context_source: Callable[[], RobotPlanningContext],
                 joints_source: Callable[[], Mapping[str, float]] | None = None,
                 link_map: Mapping[str, str] | None = None):
        """`link_map` renames the context's end-effector links to the URDF's (a simulator's meta link vs the URDF)."""
        import jax
        import jax.numpy as jnp
        import jaxlie
        import jaxls
        import pyroki as pk

        self._jnp = jnp
        self.urdf = urdf if isinstance(urdf, yourdfpy.URDF) else yourdfpy.URDF.load(str(urdf), load_meshes=False)
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.names = list(self.robot.joints.actuated_names)
        self.context_source = context_source
        self.joints_source = joints_source
        self.link_map = dict(link_map or {})
        self._lock = threading.Lock()
        self._weights: dict[tuple[str, ...], object] = {}

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

    def _rest_weight(self, arm_joints: tuple[str, ...]):
        if arm_joints not in self._weights:
            w = np.full(len(self.names), REST_WEIGHT)
            for n in arm_joints:
                w[self.names.index(n)] = 0.001
            self._weights[arm_joints] = self._jnp.asarray(w)
        return self._weights[arm_joints]

    def solve_ik(self, target_pose: Pose, robot_state: RobotState, *, arm: str = "primary", **_) -> IKResult:
        ctx = self.context_source()
        emb = robot_state.embodiment
        if arm not in ctx.joint_names:
            return IKResult(ok=False, arm=arm, embodiment=emb, error=ApiError(ErrorCode.NOT_FOUND, f"unknown arm {arm!r}"))
        if target_pose.frame != robot_state.base_frame:
            return IKResult(ok=False, arm=arm, embodiment=emb,
                            error=ApiError(ErrorCode.INVALID_REQUEST, f"target frame must be {robot_state.base_frame!r}"))
        arm_joints = tuple(ctx.joint_names[arm])
        link = self.link_map.get(ctx.end_effector_links[arm], ctx.end_effector_links[arm])
        if link not in self.robot.links.names or any(n not in self.names for n in arm_joints):
            return IKResult(ok=False, arm=arm, embodiment=emb,
                            error=ApiError(ErrorCode.UNSUPPORTED, f"URDF lacks {link!r} or the joints of {arm!r}"))
        # the pose in the URDF root frame: the context's base transform maps root -> base frame
        base_from_root = np.asarray(ctx.base_transforms[arm], dtype=float)
        root_from_base = np.linalg.inv(base_from_root)
        t = np.eye(4)
        t[:3, :3] = quaternion_wxyz_to_matrix(target_pose.quaternion_wxyz)
        t[:3, 3] = target_pose.position
        t = root_from_base @ t
        from cap_harness.geometry import matrix_to_quaternion_wxyz
        target_wxyz, target_xyz = matrix_to_quaternion_wxyz(t[:3, :3]), t[:3, 3]

        seed = np.zeros(len(self.names))
        for n, v in (self.joints_source() if self.joints_source else {}).items():
            if n in self.names:
                seed[self.names.index(n)] = v
        for a, names in ctx.joint_names.items():
            if a in robot_state.joint_positions:
                for n, v in zip(names, robot_state.joint_positions[a]):
                    if n in self.names:
                        seed[self.names.index(n)] = v
        jnp = self._jnp
        link_index = self.robot.links.names.index(link)
        with self._lock:
            sol = np.asarray(self._solve(self.robot, jnp.asarray(link_index), jnp.asarray(target_wxyz),
                                         jnp.asarray(target_xyz), jnp.asarray(seed), self._rest_weight(arm_joints)))
            pose = np.asarray(self.robot.forward_kinematics(jnp.asarray(sol))[link_index])
        joints = np.array([sol[self.names.index(n)] for n in arm_joints], dtype=float)
        pos_err = float(np.linalg.norm(pose[4:] - target_xyz))
        ori_err = float(2 * np.arccos(np.clip(abs(np.dot(pose[:4], target_wxyz)), 0, 1)))
        diag = {"position_error_m": pos_err, "orientation_error_rad": ori_err, "provider": "pyroki-urdf"}
        if pos_err > POSITION_TOLERANCE_M or ori_err > ORIENTATION_TOLERANCE_RAD:
            return IKResult(ok=False, arm=arm, embodiment=emb, diagnostics=diag,
                            error=ApiError(ErrorCode.IK_UNREACHABLE,
                                           f"IK residual {pos_err * 100:.1f} cm, {np.degrees(ori_err):.1f} deg"))
        return IKResult(ok=True, joint_positions=joints, arm=arm, embodiment=emb, diagnostics=diag)
