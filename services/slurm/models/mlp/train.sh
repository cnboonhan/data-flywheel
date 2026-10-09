# MLP behaviour-cloning baseline, and the template for a model that reads the hdf5 mix itself: copy this folder,
# replace train.py's DATA/MODEL/LOOP, point ENV at an env with its packages (torch, h5py, mlflow, pyyaml).
ENV=act

train() {
  python "$MODEL_DIR/train.py" --model "${MODEL^^}" --embodiment "$EMB_NAME" --data "$(IFS=,; echo "${DATA[*]}")" \
    ${MIX:+--mix "$MIX"} --action "$ACTION" --seed "$SEED" "${EXTRA[@]}"
}
