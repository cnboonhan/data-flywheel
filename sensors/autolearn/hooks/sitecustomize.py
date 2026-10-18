"""Isaac Sim without the crash reporter, for Genie Sim's engine (which passes SimulationApp no extra args).

Put this directory first on PYTHONPATH of any process that starts Isaac Sim (the ROS 2 launch inherits it):
SimulationApp's defaults are patched when its module is imported, so no upstream file changes.
"""

import importlib.abc
import importlib.util
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
