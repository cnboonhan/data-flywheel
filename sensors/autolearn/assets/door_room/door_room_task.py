"""The door room's BDDL activity, registered at runtime, so the bddl package and BEHAVIOR's data stay as published.

    import door_room_task
    door_room_task.register()     # before og.Environment: BehaviorTask checks the activity against the knowledge base
    cfg["scene"].update(door_room_task.SCENE)
    cfg["task"] = {"type": "BehaviorTask", "activity_name": "opening_door", **door_room_task.TASK}

opening_door (opening_door.bddl): one door, closed at the start and in the corridor (the side its leaf swings into),
the agent on the corridor floor; the goal is (open door), which OmniGibson's Open state reads true once the leaf is
past ~0.1 rad (assets/latch_door). The task instance pre-sampled by sample_door_room_task.py sits beside the scene
(scenes/door_room/json/), loaded through scene_file, as BEHAVIOR's own loader only looks in its challenge dataset.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "flywheel-assets"
SCENE_DIR = Path("/tier1/mistral/flywheel/autolearn/behavior-data") / DATASET / "scenes" / "door_room"
INSTANCE = SCENE_DIR / "json" / "door_room_task_opening_door_0_0_template.json"
SCENE = {"scene_model": "door_room", "dataset_name": DATASET, "load_room_types": None}
TASK = {"activity_definition_id": 0, "activity_instance_id": 0, "online_object_sampling": False,
        "use_presampled_robot_pose": False}
# our models, by category: BEHAVIOR's sampler only lists models from its own dataset when it matches a scene's objects
# to the activity's (bddl_utils, get_all_object_category_models_with_abilities)
MODELS = {"door": {"flylat"}, "floors": {"flyflr"}, "walls": {"flywal"}}


def register(name="opening_door", definition_id=0):
    """Add the activity to OmniGibson's knowledge base (once per process)."""
    import omnigibson.utils.bddl_utils as bu
    from omnigibson.utils.bddl_utils import get_knowledge_base

    if not getattr(bu, "_door_room_models", False):
        listed = bu.get_all_object_category_models_with_abilities

        def with_ours(category, abilities):
            return list(listed(category, abilities)) + sorted(MODELS.get(category, ()))

        bu.get_all_object_category_models_with_abilities = with_ours
        bu._door_room_models = True
    kb = get_knowledge_base()
    key = f"{name}-{definition_id}"
    if kb.get_task(key) is None:
        kb.add_task(key, (HERE / f"{name}.bddl").read_text())
    return key
