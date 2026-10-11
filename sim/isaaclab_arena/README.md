# sim/isaaclab_arena

Helpers around the [IsaacLab-Arena](../../eval/system2/README.md) submodule. Nothing here modifies the submodule; everything hooks in from outside.

| Path | Purpose |
|---|---|
| `envgen.sh` | Runs Arena's agentic environment generation (CLI runner or Streamlit GUI) through a local OpenAI-compatible proxy |
| `record.py` | `envgen.sh record`: a scripted demo of a spec, recorded from a scene camera and the wrist camera |
| `hooks/sitecustomize.py` | Import hooks loaded via `PYTHONPATH` by `envgen.sh` (so they also apply inside the GUI's separate process): a `cliproxy` inference endpoint, a `ridgeback_franka_ik` mobile manipulator, and a `splat_scene` background when `ARENA_SPLAT_SCENE` is set |

## Generate environments

Start the proxy if you need `resolve` or the GUI ([CLIProxyAPI](https://github.com/router-for-me/CLIProxyAPI), a CLI-authenticated model over the OpenAI API; `build` only needs Isaac Sim), then run `envgen.sh`. Specs go to [`eval/system2/environments/`](../../eval/system2/environments/) unless `--out_dir` is given.

```bash
export OPENAI_API_KEY=<proxy client key>
bash sim/isaaclab_arena/envgen.sh cli --mode resolve --prompt "..."   # write a YAML spec
bash sim/isaaclab_arena/envgen.sh cli --mode build --env_spec eval/system2/environments/<env>.yaml
bash sim/isaaclab_arena/envgen.sh gui                                  # live editor on http://localhost:8501
```

| Variable | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | The proxy's client key (required) | |
| `ARENA_PROXY_BASE_URL` | Proxy endpoint | `http://127.0.0.1:8317/v1` |
| `ARENA_PROXY_MODEL` | Model name the proxy serves | `claude-sonnet-5-5` |

## Splat backgrounds

Point `ARENA_SPLAT_SCENE` at a [`sim/colmap_splat`](../colmap_splat/README.md) run's `scene.usda` (trained with `--floor`, e.g. from a [`sensors/real2sim`](../../sensors/real2sim/README.md) capture), then build or resolve a spec that uses the `splat_scene` background.

```bash
export ARENA_SPLAT_SCENE=datasets/real2sim/run1/runs/colmap/<run>/scene.usda
bash sim/isaaclab_arena/envgen.sh cli --mode build --env_spec eval/system2/environments/franka_mug_to_bowl_splat_scene.yaml
bash sim/isaaclab_arena/envgen.sh cli --mode resolve --prompt "Franka at a table in the splat_scene warehouse ..."
```

Notes:
- Registered as `splat_scene` (`ARENA_SPLAT_NAME` to rename), shifted in x/y so the middle of the capture area is Arena's origin, where the robot spawns.
- The splat only renders; its baked lighting doesn't respond to the sim.
- The only collision is the floor (the capture cameras' footprint plus 2 m). Anchor on it with an object reference (`parent_id: <background id>`, `prim_path: floor`), place a table `on` it and objects on the table; an object beyond the floor fails the `on` check and falls. [`franka_mug_to_bowl_splat_scene.yaml`](../../eval/system2/environments/franka_mug_to_bowl_splat_scene.yaml) pins the table in front of the robot with `at_position`.

## Mobile manipulator

`ridgeback_franka_ik` is `franka_ik` on Isaac Lab's holonomic Clearpath Ridgeback (`isaaclab_assets` `RIDGEBACK_FRANKA_PANDA_CFG`). Its action is `franka_ik`'s 7 values (relative IK delta, gripper), then base velocity `vx, vy` (m/s) and `wz` (rad/s). The base moves on joints anchored at the spawn pose, so base velocity and IK deltas are in that fixed frame, not the robot's heading. Place the table far enough away for the base to drive up to it, as [`ridgeback_mug_to_bowl_splat_scene.yaml`](../../eval/system2/environments/ridgeback_mug_to_bowl_splat_scene.yaml) does.

## Record a demo

Run `envgen.sh record` on a spec; it writes `video_cam/clip_0000.mp4` (fixed camera at `--eye`, looking at `--target`) and `wrist_cam/clip_0000.mp4` at 15 fps.

```bash
export ARENA_SPLAT_SCENE=datasets/real2sim/run1/runs/colmap/<run>/scene.usda
bash sim/isaaclab_arena/envgen.sh record --env_spec eval/system2/environments/ridgeback_mug_to_bowl_splat_scene.yaml \
  --video_dir datasets/videos/ridgeback_demo --num_steps 460 --eye -1.7 0.8 1.6 --target 3.0 -0.1 0.6
```

Notes:
- A scripted sweep for checking a scene, not a policy: the Ridgeback drives 1.05 m to the table (`--stop_x`), the arm dips to the mug and the bowl, circles over the table, and the base backs off. Nothing is grasped; `franka_ik` specs skip the driving.
- `record.py` writes frames itself because Isaac Lab's `VideoRecorder` (moviepy 1.0.3) duplicates and drops a frame every ~7.
- Camera coordinates are Arena's (origin at the capture area's centre), so a framing fits one capture region. The one above is for `sensors/real2sim`'s `--region -9 -4 -3 2`. Keep the camera inside the captured area: floaters crowd its edges, and the floor shimmers where the capture didn't cover the viewing angle.
