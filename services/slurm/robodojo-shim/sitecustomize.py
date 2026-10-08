"""Injected into RoboDojo's eval client via PYTHONPATH by evaluate-xpolicylab.sbatch (no changes to the submodule).

On the GB300 nodes Isaac Sim 5.1 never delivers Replicator camera frames while the simulation runs on
the CPU device (RoboDojo's default), so the evaluation runs the environment on cuda:0 (env_cfg *_gpu).
RoboDojo assumes CPU tensors in a few places (np.asarray(t), t.numpy()); make those work for CUDA tensors.

Isaac Sim's SimulationApp gives the viewport about two seconds to appear. When a policy server is
initialising CUDA on the same GPU at that moment, the viewport (and its /OmniverseKit_Persp camera,
which Isaac Lab's viewport controller needs) can arrive later, and the environment fails with
"Accessed invalid null prim". Keep waiting, up to a minute.
"""
import sys

if any("eval_client" in a for a in sys.argv):
    import torch

    _numpy = torch.Tensor.numpy

    def numpy(self, *a, **k):
        return _numpy(self.detach().cpu() if self.is_cuda else self, *a, **k)

    _array = torch.Tensor.__array__

    def __array__(self, dtype=None, copy=None):
        t = self.detach().cpu() if self.is_cuda else self
        return _array(t, dtype) if dtype is not None else _array(t)

    torch.Tensor.numpy = numpy
    torch.Tensor.__array__ = __array__

    import importlib.abc, importlib.util

    def _patch_isaaclab_camera():
        """Isaac Lab's DirectRLEnv points the viewport camera at the scene on creation. RoboDojo's
        evaluation never uses that camera (its cameras are separate render products), but when the
        viewport has not produced a frame yet the camera prim is missing and Isaac Lab raises
        "Accessed invalid null prim". Make that call best-effort. Runs once Kit is up."""
        try:
            import isaaclab.sim.simulation_context as m
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[robodojo-shim] could not import isaaclab.sim.simulation_context: {e!r}\n"); return
        if getattr(m.SimulationContext, "_shim_patched", False):
            return
        orig = m.SimulationContext.set_camera_view

        def set_camera_view(self, *a, **k):
            try:
                return orig(self, *a, **k)
            except RuntimeError as e:
                if not getattr(m.SimulationContext, "_shim_warned", False):
                    m.SimulationContext._shim_warned = True
                    sys.stderr.write(f"[robodojo-shim] viewport camera unavailable ({e}); continuing without it\n"); sys.stderr.flush()

        m.SimulationContext.set_camera_view = set_camera_view
        m.SimulationContext._shim_patched = True
        sys.stderr.write("[robodojo-shim] viewport camera placement made best-effort\n"); sys.stderr.flush()

    def _patch_simulation_app(mod):
        cls = mod.SimulationApp
        orig_wait = cls._wait_for_viewport

        def wait_for_viewport(self):
            orig_wait(self)
            try:  # diagnostics
                import omni.ui, torch
                from omni.kit.viewport.utility import get_active_viewport
                vp = get_active_viewport()
                wins = sorted(w.title for w in omni.ui.Workspace.get_windows())
                pol = sorted(m for m in sys.modules if m.startswith("XPolicyLab"))
                sys.stderr.write(f"[robodojo-shim] viewport={vp} frame_info={getattr(vp, 'frame_info', None)} windows={wins} cuda_init={torch.cuda.is_initialized()} xpl_modules={pol[:8]}\n"); sys.stderr.flush()
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"[robodojo-shim] diagnostics failed: {e!r}\n")
            _patch_isaaclab_camera()
            import time
            import omni.usd
            t0 = time.time()
            while time.time() - t0 < 60:
                stage = omni.usd.get_context().get_stage()
                if stage is not None and stage.GetPrimAtPath("/OmniverseKit_Persp").IsValid():
                    return
                self.update()
            sys.stderr.write("[robodojo-shim] viewport camera still missing after 60 s\n"); sys.stderr.flush()

        cls._wait_for_viewport = wait_for_viewport

    class _Hook(importlib.abc.MetaPathFinder):
        """Patch SimulationApp once the `isaacsim` package has loaded (Isaac Lab does `from isaacsim import SimulationApp`)."""

        def find_spec(self, name, path, target=None):
            if name != "isaacsim":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            exec_module = spec.loader.exec_module

            def wrapped(m):
                exec_module(m)
                try:
                    cls = getattr(m, "SimulationApp", None)
                    mod = sys.modules.get(cls.__module__) if cls else None
                    if mod is None:
                        import isaacsim.simulation_app as mod  # noqa: PLC0415
                    _patch_simulation_app(mod)
                    sys.stderr.write("[robodojo-shim] SimulationApp viewport wait extended to 60 s\n"); sys.stderr.flush()
                except Exception as e:  # noqa: BLE001
                    sys.stderr.write(f"[robodojo-shim] viewport patch failed: {e!r}\n")
            spec.loader.exec_module = wrapped
            return spec

    sys.meta_path.insert(0, _Hook())

