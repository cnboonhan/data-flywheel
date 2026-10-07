"""Register a `cliproxy` inference endpoint in IsaacLab-Arena (loaded via PYTHONPATH by arena_envgen.sh).

Arena's endpoint presets are fixed, and its Streamlit GUI runs in a separate process, so the preset
is added with an import hook: as soon as any process imports Arena's inference_backend module,
the preset is registered. Configure with ARENA_PROXY_BASE_URL and ARENA_PROXY_MODEL.
"""

import importlib.abc
import os
import sys

_TARGET = "isaaclab_arena.agentic_environment_generation.inference_backend"


def _register(module) -> None:
    module.INFERENCE_ENDPOINTS["cliproxy"] = module.InferenceEndpoint(
        name="cliproxy",
        base_url=os.getenv("ARENA_PROXY_BASE_URL", "http://127.0.0.1:8317/v1"),
        model=os.getenv("ARENA_PROXY_MODEL", "claude-sonnet-5-5"),
        api_key_env_var="OPENAI_API_KEY",
    )


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            _register(module)

        spec.loader.exec_module = exec_module
        return spec


import importlib.util  # noqa: E402

sys.meta_path.insert(0, _Finder())
