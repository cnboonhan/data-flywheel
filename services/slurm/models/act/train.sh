# ACT (XPolicyLab policy/ACT): the policy's process_data.sh, then imitate_episodes.py as policy/ACT/train.sh runs it,
# through lib/xpolicylab_train.py for MLflow. Defaults are XPolicyLab's; override after `--`, e.g. --num_epochs 30.
ENV=act
source "$LIB/xpolicylab.sh"

train() {
  xpolicylab_single_dataset
  xpolicylab_links
  local setting="$BENCH-$TASK-$EMB_XPL_ENV_CFG-$ACTION" action_dim
  local ckpt_dir="$XPL/policy/ACT/checkpoints/$setting-$SEED"
  cd "$XPL/policy/ACT"
  bash process_data.sh "$BENCH" "$TASK" "$EMB_XPL_ENV_CFG" "$ACTION"
  action_dim=$(bash "$XPL/utils/get_action_dim.sh" "$PROJECT_ROOT" "$EMB_XPL_ENV_CFG")
  export ACT_ACTION_DIM=$action_dim
  python "$LIB/xpolicylab_train.py" --model ACT --embodiment "$EMB_NAME" --data "$BENCH/$TASK" --action "$ACTION" --seed "$SEED" \
    --ckpt-dir "$ckpt_dir" --ckpt-name "$setting-$SEED" -- \
    imitate_episodes.py --bench_name "$BENCH" --task_name "$TASK" --ckpt_setting "$setting" \
    --ckpt_dir "$ckpt_dir" --policy_class ACT --kl_weight 10 --chunk_size 50 \
    --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 --num_epochs 6000 --lr 1e-5 --save_freq 6000 --seed "$SEED" "${EXTRA[@]}"
}
