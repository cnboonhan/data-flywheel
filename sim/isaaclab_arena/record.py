"""Record a scripted demo of an Arena spec from a scene camera and the wrist camera (franka_ik, ridgeback_franka_ik).

    bash sim/isaaclab_arena/envgen.sh record --env_spec <spec.yaml> --video_dir <dir> [--num_steps 460] [--stop_x 1.05]

The robot holds its hand orientation and steps (relative IK) to hover over the task's pick object, dip, close the
gripper, move over the destination, dip, open, and circle over the two; a mobile embodiment first drives its base to
--stop_x and backs off at the end. A scripted sweep, not a policy: the gripper does not grasp. Terminations are
evaluated but never end the episode. Frames go straight to <dir>/<camera>/clip_0000.mp4: Isaac Lab's VideoRecorder
(moviepy 1.0.3) duplicates and drops a frame every ~7 at 15 fps.
"""

import sys
from pathlib import Path

sys.path.insert(0, "isaaclab_arena_examples/agentic_environment_generation")   # run from the Arena checkout
import cli_runner as cr  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402

parser = cr.get_isaaclab_arena_cli_parser()
cr.add_agentic_env_gen_runner_cli_args(parser)
parser.add_argument("--video_dir", type=Path, required=True)
parser.add_argument("--eye", type=float, nargs=3, default=[0.3, -2.4, 1.9], help="scene camera position")
parser.add_argument("--target", type=float, nargs=3, default=[1.5, 0.0, 0.6], help="scene camera look-at point")
parser.add_argument("--stop_x", type=float, default=1.05, help="base x to drive to (mobile embodiments)")
parser.set_defaults(enable_cameras=True)
args = parser.parse_args()

with SimulationAppContext(args):
    import gymnasium as gym
    import imageio.v3 as iio
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from isaaclab.sensors import CameraCfg
    from scipy.spatial.transform import Rotation

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    spec = ArenaEnvGraphSpec.from_yaml(Path(args.env_spec))
    task = spec.task.subtasks[0].params   # PickAndPlaceTask: pick_up_object, destination_location
    name, cfg, kwargs = ArenaEnvBuilder(
        spec.to_arena_env(enable_cameras=True), cr.arena_env_builder_cfg_from_argparse(args)
    ).build_registered()
    cfg.scene.wrist_cam.width, cfg.scene.wrist_cam.height = 640, 360
    d = np.subtract(args.target, args.eye)   # "world" camera convention: x forward, z up
    rot = Rotation.from_euler("ZY", [np.arctan2(d[1], d[0]), np.arctan2(-d[2], np.hypot(d[0], d[1]))]).as_quat()
    cfg.scene.video_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/VideoCam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, horizontal_aperture=20.955, clipping_range=(0.01, 1000.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(args.eye), rot=tuple(rot), convention="world"),
    )
    env = gym.make(name, cfg=cfg, **kwargs)
    u = env.unwrapped
    tm = u.termination_manager
    for t in tm.active_terms:   # keep evaluating, never terminate
        tc = tm.get_term_cfg(t)
        tc.func = (lambda f: lambda env, **kw: torch.zeros_like(f(env, **kw)))(tc.func)
        tm.set_term_cfg(t, tc)
    env.reset()

    sc, origin = u.scene, u.scene.env_origins[0]
    robot = sc["robot"]
    hand = robot.find_bodies("panda_hand")[0][0]
    mobile = "base_action" in u.action_manager.active_terms
    local = lambda n: (sc[n].data.root_pos_w[0] - origin).cpu().numpy()  # noqa: E731
    pick, dest = local(task["pick_up_object"]), local(task["destination_location"])
    hover = max(pick[2], dest[2]) + 0.18   # hand heights; the fingertips are 0.107 m lower
    c = (pick[:2] + dest[:2]) / 2
    arm = [  # (hand x, y, z), gripper (+1 open, -1 close)
        ((*pick[:2], hover), 1), ((*pick[:2], pick[2] + 0.15), 1), ((*pick[:2], pick[2] + 0.15), -1),
        ((*pick[:2], hover), -1), ((*dest[:2], hover), -1), ((*dest[:2], dest[2] + 0.13), -1),
        ((*dest[:2], dest[2] + 0.13), 1), ((*dest[:2], hover), 1),
        *[((*(c + 0.12 * np.array([np.cos(a), np.sin(a)])), hover), 1) for a in np.linspace(0, 2 * np.pi, 13)],
    ]
    plan = [("arm", w) for w in arm]
    if mobile:   # base joints are anchored at the spawn pose: x, y, yaw in that frame
        plan = [("base", (args.stop_x, 0.0, 0.0))] + plan + [("base", (args.stop_x - 0.6, 0.0, 0.8))]

    scale = cfg.actions.arm_action.scale
    quat = lambda: Rotation.from_quat(robot.data.body_quat_w[0, hand].cpu().numpy())  # noqa: E731  (x, y, z, w)
    q_hold = quat()
    frames = {"video_cam": [], "wrist_cam": []}
    wp, held = 0, 0
    for step in range(args.num_steps):
        kind, target = plan[min(wp, len(plan) - 1)]
        a = torch.zeros(u.action_space.shape, device=u.device)
        a[0, 3:6] = torch.tensor(np.clip((q_hold * quat().inv()).as_rotvec(), -0.1, 0.1) / scale)
        if kind == "base":
            err = np.subtract(target, robot.data.joint_pos[0, :3].cpu().numpy())
            a[0, 7:] = torch.tensor(np.clip(1.5 * err, -0.4, 0.4))
            a[0, 6] = 1
            done, timeout = np.abs(err).max() < 0.01, 150
            if wp == len(plan) - 1:
                q_hold = quat()   # the hand turns with the base
        else:
            err = np.asarray(target[0]) - (robot.data.body_pos_w[0, hand] - origin).cpu().numpy()
            a[0, :3] = torch.tensor(np.clip(err, -0.02, 0.02) / scale)
            a[0, 6] = target[1]
            done, timeout = np.linalg.norm(err) < 0.015, 60
        env.step(a)
        for cam, f in frames.items():
            f.append(sc[cam].data.output["rgb"][0, ..., :3].cpu().numpy())
        held += 1
        if wp < len(plan) and (done or held > timeout):
            print(f"[record] step {step}: {kind} waypoint {wp} {'reached' if done else 'timed out'}", flush=True)
            wp, held = wp + 1, 0

    for cam, f in frames.items():
        out = args.video_dir / cam / "clip_0000.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(out, np.stack(f), fps=round(1 / u.step_dt), codec="libx264", quality=8)
        print(f"[record] {out}", flush=True)
    env.close()
