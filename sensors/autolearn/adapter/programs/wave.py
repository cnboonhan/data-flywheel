# Smoke program for any ASPIRE environment: lift the right gripper 8 cm, close it, open it. No perception.
state = get_robot_state()
arm = "secondary"
start = state.end_effector_poses[arm]
log = {"start": [float(v) for v in start.position]}
up = {"position": [float(start.position[0]), float(start.position[1]), float(start.position[2]) + 0.08],
      "quaternion_wxyz": [float(v) for v in start.quaternion_wxyz], "frame": start.frame}
r = move_to_pose(up, arm=arm, tolerance=0.01, max_steps=90)
log["move_ok"] = bool(r.ok)
log["move_steps"] = int(r.steps_executed)
log["move_error"] = None if r.error is None else str(r.error.message)
after = get_robot_state().end_effector_poses[arm]
log["after"] = [float(v) for v in after.position]
log["rise_m"] = float(after.position[2] - start.position[2])
g1 = close_gripper(arm=arm)
g2 = open_gripper(arm=arm)
log["gripper_ok"] = bool(g1.ok and g2.ok)
log["gripper_after"] = float(get_robot_state().gripper_positions[arm])
print(log)
result = log
