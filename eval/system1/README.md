# eval/system1

VLA benchmarks for action models (system 1). Both are git submodules; see each one's upstream README for installation.

| Benchmark | Path | Simulator | Env |
|---|---|---|---|
| [RoboDojo](https://github.com/robodojo-benchmark/RoboDojo) | `RoboDojo` | Isaac Sim 5.1 | conda `RoboDojo` |
| [RoboTwin 2.0](https://github.com/RoboTwin-Platform/RoboTwin) | `RoboTwin` | SAPIEN | conda `RoboTwin` |

Policies plug in through [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) at `XPolicyLab/policy/<POLICY>/` inside each benchmark.

**Gotchas**
- **Submodule versions:** the RoboDojo and RoboTwin installers move XPolicyLab to its latest commit. Run `git submodule update` inside the benchmark afterwards to go back to the pinned version.
