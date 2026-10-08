# xpolicylab

Python the Slurm jobs run ([slurm/](../slurm/README.md)); seeded into the `admin/pipelines` repo. Conversions into `processed/xpolicylab/` live in [gitea/adapter/](../gitea/README.md).

| Script | Does |
|---|---|
| `train_mlflow.py` | wraps a policy's training command: per-epoch losses to MLflow (experiment `xpolicylab`, run `<policy>-<bench>-<task>-<env_cfg>-<action>-<seed>`), checkpoints as artifacts, registered model `<policy>-<bench>-<task>` |
| `eval_mlflow.py` | RoboDojo `_result.json` → `eval/success_rate`, `eval/score`, `eval/episodes` and the videos on the training run; tags the model version |

```bash
python eval_mlflow.py --policy ACT --bench RoboDojo --task stack_bowls --ckpt RoboDojo-stack_bowls-arx_x5-joint-0 --env-cfg arx_x5_gpu --action joint --seed 0 --result-root $ROBODOJO_DIR/eval_result --since <epoch>
```

Supported policies: `ACT`, `DP`. `env_cfg=arx_x5` is a stand-in for Galaxea's r1lite (same 14-D layout); a proper `r1lite` entry needs a RoboDojo/XPolicyLab fork.
