"""Isaac Sim without the crash reporter, for Genie Sim's engine (which passes SimulationApp no extra args).

Put this directory first on PYTHONPATH of any process that starts Isaac Sim (the ROS 2 launch inherits it):
SimulationApp's defaults are patched when its module is imported, so no upstream file changes.
"""

import importlib.abc
import importlib.util
import sys

_MODULE = "isaacsim"   # its __init__ imports SimulationApp (the engine does `from isaacsim import SimulationApp`)


class _PatchSimulationApp(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _MODULE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def patched(module):
            exec_module(module)
            cfg = module.SimulationApp.DEFAULT_LAUNCHER_CONFIG
            cfg["enable_crashreporter"] = False
            cfg["extra_args"] = list(cfg.get("extra_args", [])) + ["--/crashreporter/enabled=false"]

        spec.loader.exec_module = patched
        return spec


sys.meta_path.insert(0, _PatchSimulationApp())
