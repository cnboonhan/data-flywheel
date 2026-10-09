# DP in RoboDojo through XPolicyLab's policy/DP/eval.sh (lib/xpolicylab.sh), result on the training run in MLflow.
ENV=dp   # mlflow for eval_mlflow.py; eval.sh itself activates the dp and robodojo envs through the conda shim
source "$LIB/xpolicylab.sh"

evaluate() {
  xpolicylab_eval DP
}
