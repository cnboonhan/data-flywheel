"""Write instructions.json and problems.json for every instance, in the shipped
llm_task/place_object_relative_position format, from spec.py plus the object ids the generator assigned
(scene_info.json's layout keys). The LLM-backed generators (evaluator/generators/*.py) need an API key;
this is the same content by hand.

usage: python3 checks.py <llm_task/<sub_task> dir>
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spec import ALIGN_AXIS, ALIGN_TOL, FOLLOW_BBOX, FOLLOW_STEPS, PLACE_STEPS, SPEC  # noqa: E402

SUB_TASK = "place_object_relative_position_lab"


def object_id(layout, asset_id):
    ids = [k for k, v in layout.items() if v["usd"] == asset_id]
    if len(ids) != 1:
        raise SystemExit(f"expected exactly one {asset_id} in the layout, found {ids}")
    return ids[0]


def problems(active, passive, relation):
    return {
        "problem1": {
            "Acts": [
                {
                    "ActionList": [
                        {
                            "ActionSetWaitAny": [
                                {"Follow": f"{active}|{FOLLOW_BBOX}|left_gripper"},
                                {"Follow": f"{active}|{FOLLOW_BBOX}|right_gripper"},
                                {"StepOut": FOLLOW_STEPS},
                            ]
                        },
                        {
                            "ActionSetWaitAny": [
                                {
                                    "MixedRules": {
                                        "check_interval": 1,
                                        "mode": "and",
                                        "rules": [
                                            {"name": "RelativePosition", "params": f"{active}|{passive}|{relation}"},
                                            {
                                                "name": "RelativePosition",
                                                "params": f"{active}|{passive}|{ALIGN_AXIS[relation]}|{ALIGN_TOL}",
                                            },
                                        ],
                                    }
                                },
                                {"StepOut": PLACE_STEPS},
                            ]
                        },
                    ]
                }
            ],
            "Init": [],
            "Objects": [],
            "Problem": SUB_TASK,
        }
    }


def main(root):
    for n, task in enumerate(SPEC):
        inst = os.path.join(root, str(n))
        with open(os.path.join(inst, "scene_info.json")) as f:
            layout = json.load(f)["layout"]
        active, passive = object_id(layout, task["active"]), object_id(layout, task["passive"])
        with open(os.path.join(inst, "instructions.json"), "w") as f:
            json.dump({"instructions": [{"instruction": task["instruction"]}], "task_id": f"{SUB_TASK}_{n}"}, f, indent=2)
            f.write("\n")
        with open(os.path.join(inst, "problems.json"), "w") as f:
            json.dump(problems(active, passive, task["relation"]), f, indent=2)
            f.write("\n")
        print(f"{n}: {task['instruction']!r}  {active} {task['relation']} {passive}")


if __name__ == "__main__":
    main(sys.argv[1])
