# Diffusion Policy (XPolicyLab policy/DP): process_data.sh (zarr), then train.py with the robot_dp config,
# through lib/xpolicylab_train.py for MLflow. Hydra overrides after `--`, e.g. training.num_epochs=30.
ENV=dp
source "$LIB/xpolicylab.sh"

train() {
  xpolicylab_single_dataset
  xpolicylab_links
  local setting="$BENCH-$TASK-$EMB_XPL_ENV_CFG-$ACTION" action_dim
  local ckpt_dir="$XPL/policy/DP/checkpoints/$setting-$SEED"   # robotworkspace.py saves there regardless of cwd
  cd "$XPL/policy/DP"
  bash process_data.sh "$BENCH" "$TASK" "$EMB_XPL_ENV_CFG" "$ACTION"
  action_dim=$(bash "$XPL/utils/get_action_dim.sh" "$PROJECT_ROOT" "$EMB_XPL_ENV_CFG")
  export HYDRA_FULL_ERROR=1
  python "$LIB/xpolicylab_train.py" --model DP --embodiment "$EMB_NAME" --data "$BENCH/$TASK" --action "$ACTION" --seed "$SEED" \
    --ckpt-dir "$ckpt_dir" --ckpt-name "$setting-$SEED" -- \
    train.py --config-name=robot_dp.yaml bench_name="$BENCH" task.name="$TASK" \
    "task.shape_meta.action.shape=[$action_dim]" \
    "+task.shape_meta.obs.agent_pos.shape=[$action_dim]" "+task.shape_meta.obs.agent_pos.type=low_dim" \
    task.dataset.zarr_path="data/$setting.zarr" training.debug=False training.seed="$SEED" training.device=cuda:0 \
    exp_name="$TASK-robot_dp-train" logging.mode=offline setting="$EMB_XPL_ENV_CFG" "${EXTRA[@]}"
}
