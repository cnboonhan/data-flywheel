"""Isaac Sim without the crash reporter, for Genie Sim's engine (which passes SimulationApp no extra args), and
benchmark tasks added beside upstream (adapter/tasks/) registered in the benchmark's code tables.

Put this directory first on PYTHONPATH of any process that starts Isaac Sim (the ROS 2 launch inherits it):
SimulationApp's defaults are patched when its module is imported, so no upstream file changes. Likewise a task
folder `llm_task/<sub_task>/task_info.json` with `{"like": "<shipped sub_task>"}` gets that task's robot init
states (`robot_init_states.TASK_INFO_DICT`, without which the env has no `init_arm`) and scoring steps
(`eval_utils.TASK_STEPS`/`TASK_SUBTASK_STEPS`) when those modules are imported.
"""

import glob
import importlib.abc
import importlib.util
import json
import os
import sys

# The pip install defines SimulationApp in `isaacsim` itself; the /isaac-sim install (a namespace package) in
# `isaacsim.simulation_app`. Patch whichever of these first carries the class.
_MODULES = {"isaacsim", "isaacsim.simulation_app", "isaacsim.simulation_app.simulation_app"}
_FLAG = "--/crashreporter/enabled=false"


class _PatchSimulationApp(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _MODULES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def patched(module):
            exec_module(module)
            cfg = getattr(getattr(module, "SimulationApp", None), "DEFAULT_LAUNCHER_CONFIG", None)
            if isinstance(cfg, dict) and _FLAG not in cfg.get("extra_args", []):
                cfg["enable_crashreporter"] = False
                cfg["extra_args"] = list(cfg.get("extra_args", [])) + [_FLAG]

        spec.loader.exec_module = patched
        return spec


sys.meta_path.insert(0, _PatchSimulationApp())

_TABLES = {  # module -> (dict names to extend, path from the module's file to the package root)
    "geniesim_benchmark.benchmark.config.robot_init_states": (("TASK_INFO_DICT",), "../.."),
    "geniesim_benchmark.plugins.output_system.eval_utils": (("TASK_STEPS", "TASK_SUBTASK_STEPS"), "../.."),
}


def _added_tasks(pkg_root):
    for f in glob.glob(os.path.join(pkg_root, "benchmark/config/llm_task/*/task_info.json")):
        with open(f) as fh:
            yield os.path.basename(os.path.dirname(f)), json.load(fh)


class _RegisterAddedTasks(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TABLES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module
        tables, up = _TABLES[name]

        def patched(module):
            exec_module(module)
            root = os.path.normpath(os.path.join(os.path.dirname(module.__file__), up))
            for task, info in _added_tasks(root):
                for table in tables:
                    d = getattr(module, table, None)
                    if isinstance(d, dict) and info.get("like") in d and task not in d:
                        d[task] = d[info["like"]]

        spec.loader.exec_module = patched
        return spec


sys.meta_path.insert(0, _RegisterAddedTasks())
