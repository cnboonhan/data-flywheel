"""Run one of openpi's scripts with the R1 Pro configs (r1pro.py) registered.

    venv-pi05/bin/python run.py train pi05_r1pro_picking_up_cup --exp-name <run> [openpi train flags...]
    venv-pi05/bin/python run.py compute_norm_stats --config-name pi05_r1pro_picking_up_cup
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OPENPI = HERE.parents[3] / "eval/system1/RoboDojo/XPolicyLab/policy/Pi_05/openpi"
sys.path.insert(0, str(HERE))

import r1pro  # noqa: E402

r1pro.register()
script = OPENPI / "scripts" / f"{sys.argv[1]}.py"
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
