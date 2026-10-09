# sim/isaaclab_arena

Helpers around the [IsaacLab-Arena](../../eval/system2/README.md) submodule. Nothing here modifies the submodule; everything hooks in from outside.

| Path | Purpose |
|---|---|
| `envgen.sh` | Runs Arena's agentic environment generation (CLI runner or Streamlit GUI) through a local OpenAI-compatible proxy |
| `record.py` | `envgen.sh record`: a scripted demo of a spec, recorded from a scene camera and the wrist camera |
| `hooks/sitecustomize.py` | Import hooks loaded via `PYTHONPATH` by `envgen.sh` (so they also apply inside the GUI's separate process): a `cliproxy` inference endpoint, a `ridgeback_franka_ik` mobile manipulator, and a `splat_scene` background when `ARENA_SPLAT_SCENE` is set |

## Generate environments

```bash
bash sim/isaaclab_arena/envgen.sh cli --mode resolve --prompt "..."   # write a YAML spec
bash sim/isaaclab_arena/envgen.sh cli --mode build --env_spec eval/system2/environments/<env>.yaml
bash sim/isaaclab_arena/envgen.sh gui                                  # live editor on http://localhost:8501
```

Specs go to [`eval/system2/environments/`](../../eval/system2/environments/) unless `--out_dir` is given.

Environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | The proxy's client key (required) | |
| `ARENA_PROXY_BASE_URL` | Proxy endpoint | `http://127.0.0.1:8317/v1` |
| `ARENA_PROXY_MODEL` | Model name the proxy serves | `claude-sonnet-5-5` |

The proxy is [CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI), which exposes a CLI-authenticated model over the OpenAI API. `build` mode only needs Isaac Sim (it instantiates an existing spec); `resolve` and the GUI need the proxy running.

## Splat backgrounds

A [`sim/splat`](../splat/README.md) run trained with `--floor` (e.g. from a [`sensors/real2sim`](../../sensors/real2sim/README.md) capture) can be the background of an Arena environment:

```bash
export ARENA_SPLAT_SCENE=datasets/real2sim/run1/runs/colmap/<run>/scene.usda
bash sim/isaaclab_arena/envgen.sh cli --mode build --env_spec eval/system2/environments/franka_mug_to_bowl_splat_scene.yaml
bash sim/isaaclab_arena/envgen.sh cli --mode resolve --prompt "Franka at a table in the splat_scene warehouse ..."
```

The hook registers it as `splat_scene` (`ARENA_SPLAT_NAME` to rename), shifted in x/y so the middle of the capture area is Arena's origin, where the robot spawns. The splat only renders, and its baked lighting doesn't respond to the sim. The only collision is the floor, so anchor on it with an object reference (`parent_id: <background id>`, `prim_path: floor`), then place real assets such as a table `on` it and objects on the table. The floor covers the capture cameras' footprint plus 2 m; an object placed beyond it fails the `on` check and falls. [`franka_mug_to_bowl_splat_scene.yaml`](../../eval/system2/environments/franka_mug_to_bowl_splat_scene.yaml) pins the table in front of the robot with `at_position`.

## Mobile manipulator

`ridgeback_franka_ik` is `franka_ik` on Isaac Lab's holonomic Clearpath Ridgeback (`isaaclab_assets` `RIDGEBACK_FRANKA_PANDA_CFG`). Its action is `franka_ik`'s 7 values (relative IK delta, gripper), then base velocity `vx, vy` (m/s) and `wz` (rad/s). The base moves on joints anchored at the spawn pose, so base velocity and IK deltas are in that fixed frame, not the robot's heading. Place the table far enough away for the base to drive up to it, as [`ridgeback_mug_to_bowl_splat_scene.yaml`](../../eval/system2/environments/ridgeback_mug_to_bowl_splat_scene.yaml) does.

## Record a demo

```bash
export ARENA_SPLAT_SCENE=datasets/real2sim/run1/runs/colmap/<run>/scene.usda
bash sim/isaaclab_arena/envgen.sh record --env_spec eval/system2/environments/ridgeback_mug_to_bowl_splat_scene.yaml \
  --video_dir datasets/videos/ridgeback_demo --num_steps 460
```

This writes `video_cam/clip_0000.mp4` (a fixed camera, `--eye`/`--target`) and `wrist_cam/clip_0000.mp4` at 15 fps: the
Ridgeback drives 1.05 m to the table (`--stop_x`), the arm hovers over and dips to the mug, moves over the bowl and
dips, circles over the table, and the base backs off and turns. It's a scripted sweep for checking a scene, not a
policy, so nothing is grasped; `franka_ik` specs skip the driving. `record.py` writes the frames itself because Isaac
Lab's `VideoRecorder` (moviepy 1.0.3) duplicates and drops a frame every ~7. In a splat scene, the floor shimmers in
the wrist view where it's seen from angles the capture didn't cover.
