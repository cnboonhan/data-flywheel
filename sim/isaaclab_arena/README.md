# sim/isaaclab_arena

Helpers around the [IsaacLab-Arena](../../eval/system2/README.md) submodule. Nothing here modifies the submodule; everything hooks in from outside.

| Path | Purpose |
|---|---|
| `envgen.sh` | Runs Arena's agentic environment generation (CLI runner or Streamlit GUI) through a local OpenAI-compatible proxy |
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

The hook registers it as `splat_scene` (`ARENA_SPLAT_NAME` to rename), shifted in x/y so the middle of the capture area is Arena's origin, where the robot spawns. The splat only renders, and its baked lighting doesn't respond to the sim. The only collision is the floor, so anchor on it with an object reference (`parent_id: <background id>`, `prim_path: floor`), then place real assets such as a table `on` it and objects on the table. [`franka_mug_to_bowl_splat_scene.yaml`](../../eval/system2/environments/franka_mug_to_bowl_splat_scene.yaml) pins the table in front of the robot with `at_position`.

## Mobile manipulator

`ridgeback_franka_ik` is `franka_ik` on Isaac Lab's holonomic Clearpath Ridgeback (`isaaclab_assets` `RIDGEBACK_FRANKA_PANDA_CFG`). Its action is `franka_ik`'s 7 values (relative IK delta, gripper), then base velocity `vx, vy` (m/s) and `wz` (rad/s). The base moves on joints anchored at the spawn pose, so base velocity and IK deltas are in that fixed frame, not the robot's heading. Place the table far enough away for the base to drive up to it.
