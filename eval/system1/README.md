# eval/system1

The VLA benchmark for action models (system 1), a git submodule. Install it per its upstream README; on the cluster, `setup-envs` builds the RoboDojo env ([gitea/setup/](../../services/gitea/setup/README.md)) and Slurm jobs run it ([slurm/](../../services/slurm/README.md)).

| Benchmark | Path | Simulator | Env |
|---|---|---|---|
| [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) | `RoboDojo` | Isaac Sim 5.1 | conda `RoboDojo` |

Policies plug in through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) at `XPolicyLab/policy/<POLICY>/` inside it.

**Notes**
- **Submodule versions:** the RoboDojo installer moves XPolicyLab to its latest commit. Go back to the pinned version afterwards:
  ```bash
  git -C eval/system1/RoboDojo submodule update
  ```
