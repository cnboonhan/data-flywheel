# eval/system1

VLA benchmarks for action models (system 1), both git submodules. Install each per its upstream README; on the cluster, `setup-envs` builds the RoboDojo env ([gitea/setup/](../../services/gitea/setup/README.md)) and Slurm jobs run it ([slurm/](../../services/slurm/README.md)).

| Benchmark | Path | Simulator | Env |
|---|---|---|---|
| [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) | `RoboDojo` | Isaac Sim 5.1 | conda `RoboDojo` |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) | `RoboTwin` | SAPIEN | conda `RoboTwin` |

Policies plug in through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) at `XPolicyLab/policy/<POLICY>/` inside each benchmark.

**Notes**
- **Submodule versions:** the RoboDojo and RoboTwin installers move XPolicyLab to its latest commit. Go back to the pinned version afterwards:
  ```bash
  git -C eval/system1/RoboDojo submodule update   # or eval/system1/RoboTwin
  ```
