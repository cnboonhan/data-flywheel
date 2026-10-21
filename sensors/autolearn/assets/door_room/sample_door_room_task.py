"""Sample the opening_door task instance on the door room and save it beside the scene; check the goal both ways.

    venv-behavior/bin/python sample_door_room_task.py            # online sampling, then save
    venv-behavior/bin/python sample_door_room_task.py --check    # load the saved instance, check (open door)

The R1 Pro uses ASPIRE's robot config (cap_harness.behavior.config). With online sampling the door and the floor are
matched to the scene's own objects (door.n.01 -> door_flylat_0, floor.n.01 -> floors_flyflr_0, both in corridor_0);
nothing is placed. The goal check is BehaviorTask's own: closed -> unmet, the leaf teleported to 1.2 rad -> met.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent / "aspire" / "src")]
p = argparse.ArgumentParser()
p.add_argument("--check", action="store_true")
args = p.parse_args()

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

gm.ENABLE_OBJECT_STATES = True
og.launch()
import carb  # noqa: E402

carb.settings.get_settings().set("/crashreporter/enabled", False)
import door_room_task  # noqa: E402
from cap_harness.behavior.config import build_environment_config  # noqa: E402
from cap_harness.behavior.registry import BehaviorTaskMetadata  # noqa: E402

door_room_task.register()
meta = BehaviorTaskMetadata(task_name="opening_door", activity_name="opening_door", language="open the door",
                            target_scope="door.n.01_1", target_prompt="door handle", support_prompt="door",
                            scene_model="door_room", load_room_types=("corridor", "entryway"))
cfg = build_environment_config(meta, camera_width=640, camera_height=480, horizon=1000)
cfg["scene"].update(door_room_task.SCENE)
cfg["task"].update(door_room_task.TASK)
if args.check:
    cfg["scene"]["scene_file"] = str(door_room_task.INSTANCE)
else:
    cfg["task"]["online_object_sampling"] = True
env = og.Environment(configs=cfg)
env.reset()
task = env.task
print("scope", {k: (v.name if v is not None and hasattr(v, "name") else v) for k, v in task.object_scope.items()}, flush=True)


def goal():
    done, status = task.compiled_task.check_goal(task._evaluate_predicate)
    return done, status


door = task.object_scope["door.n.01_1"]
leaf = door.joints["j_leaf"]
print("closed: goal met", goal()[0], flush=True)
leaf.set_pos(1.2)
for _ in range(5):
    og.sim.step()
print("leaf at 1.2 rad: goal met", goal()[0], flush=True)
leaf.set_pos(0.0)
for _ in range(5):
    og.sim.step()
if not args.check:
    task.save_task(env, save_dir=str(door_room_task.INSTANCE.parent), override=True)
    print("saved", door_room_task.INSTANCE, door_room_task.INSTANCE.exists(), flush=True)
og.shutdown()
