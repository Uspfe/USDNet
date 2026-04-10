#!/bin/bash
set -euo pipefail

# Run movable-part articulation inference on one custom NPZ (xyz/rgb)
# using the same main pipeline as scripts/infer_mov.sh.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_npz> [save_dir] [custom_data_dir]"
  exit 1
fi

INPUT_NPZ="$1"
SAVE_DIR="${2:-./results/inference_mov_custom_npz}"
CUSTOM_DATA_DIR="${3:-./data/processed/custom_mov_npz}"

if [[ ! -f "$INPUT_NPZ" ]]; then
  echo "Input NPZ not found: $INPUT_NPZ"
  exit 1
fi

export WANDB_MODE="offline"
export OMP_NUM_THREADS=3
export CUDA_LAUNCH_BLOCKING=1

CURR_QUERY=100
BACKBONE_CKPT="./checkpoints/mov_trainval.ckpt"

mkdir -p "$SAVE_DIR"

pixi run python tools/prepare_custom_npz_for_mov_inference.py \
  --input_npz "$INPUT_NPZ" \
  --output_root "$CUSTOM_DATA_DIR"

pixi run python main_instance_segmentation_articulation.py \
  general.experiment_name="inference_mov_custom_npz" \
  general.project_name="articulate3d_custom" \
  data/datasets=custom_mov_npz \
  general.num_targets=4 \
  general.eval_on_segments=false \
  general.train_on_segments=false \
  general.save_dir="$SAVE_DIR" \
  general.eval_articulation=true \
  general.eval_hierarchy_inter=false \
  general.debug=true \
  general.train_mode=false \
  general.checkpoint="$BACKBONE_CKPT" \
  data.train_mode="train_validation" \
  data.validation_mode="validation" \
  data.test_mode="test" \
  data.num_labels=3 \
  data.batch_size=1 \
  data.voxel_size=0.02 \
  data.load_articulation=false \
  data.use_hierarchy=false \
  data.cropping=false \
  data.crop_length=8.0 \
  data.crop_min_size=100000 \
  data.use_coarse_to_fine=true \
  data.c2f_rad=0.1 \
  data.c2f_decay=0.4 \
  data.c2f_alpha=100 \
  model.num_queries=${CURR_QUERY} \
  model.predict_articulation_mode=2 \
  model.predict_hierarchy_interaction=false \
  model.predict_articulation=true \
  loss.regular_arti_loss=false \
  loss.losses="[labels,masks, articulations]" \
  trainer.check_val_every_n_epoch=20 \
  optimizer.lr=0.0001

PREDS_PKL="$SAVE_DIR/debug/val_preds/preds.pkl"
if [[ ! -f "$PREDS_PKL" ]]; then
  echo "Expected predictions file not found: $PREDS_PKL"
  exit 1
fi

echo "Raw predictions saved to: $PREDS_PKL"
