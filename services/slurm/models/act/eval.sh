# ACT in RoboDojo through XPolicyLab's policy/ACT/eval.sh (lib/xpolicylab.sh), result on the training run in MLflow.
ENV=act   # mlflow for eval_mlflow.py; eval.sh itself activates the act and robodojo envs through the conda shim
source "$LIB/xpolicylab.sh"

evaluate() {
  xpolicylab_eval ACT
}
