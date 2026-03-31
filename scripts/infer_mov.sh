#!/bin/bash
set -euo pipefail

# inference the movable part segmentation and articulation prediction

# cd USDNet # TODO: change the path to the USDNet directory

# NOTE:
# 1. Please check TODOs before running the script
# 2. general.debug is set to true to save the predictions in pickle format

export WANDB_API_KEY="your wandb api key for log"
export WANDB_MODE="online"
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_LAUNCH_BLOCKING=1
GPUS=1
CURR_DBSCAN=0.95
CURR_TOPK=150
CURR_QUERY=100

BACKBONE_CKPT="mov_trainval.ckpt"
SAVE_DIR=./results/inference_mov_articulation 
DATA_DB=./data/processed/articulate3d_challenge_mov/test_database.yaml

mkdir -p $SAVE_DIR

# inference to get the predictions
python main_instance_segmentation_articulation.py \
general.experiment_name="inference_mov_articulation" \
general.project_name="articulate3d_challenge" \
data/datasets=articulate3d_challenge_mov \
general.num_targets=4 \
general.eval_on_segments=false \
general.train_on_segments=false \
general.save_dir=$SAVE_DIR \
general.eval_articulation=true \
general.eval_hierarchy_inter=false \
general.debug=true \
general.train_mode=false \
general.checkpoint=$BACKBONE_CKPT \
data.train_mode="train_validation" \
data.num_labels=3 \
data.batch_size=1 \
data.voxel_size=0.02 \
data.load_articulation=true \
data.use_hierarchy=false \
data.cropping=false \
data.crop_length=8.0 \
data.crop_min_size=100000 \
data.use_coarse_to_fine=true \
data.c2f_rad=0.1 \
data.c2f_decay=0.4 \
data.c2f_alpha=100 \
data.test_mode="validation" \
model.num_queries=${CURR_QUERY} \
model.predict_articulation_mode=2 \
model.predict_hierarchy_interaction=false \
model.predict_articulation=true \
loss.regular_arti_loss=false \
loss.losses="[labels,masks, articulations]" \
trainer.check_val_every_n_epoch=20 \
optimizer.lr=0.0001 \

PREDS_PKL="$SAVE_DIR/debug/val_preds/preds.pkl"
if [ ! -f "$PREDS_PKL" ]; then
	echo "Expected predictions file not found: $PREDS_PKL"
	echo "Inference did not produce outputs; check earlier errors in logs."
	exit 1
fi

# trainsfer the prediction from .pickle to the format of the scannet 3D instance segmentation in .zip to save space for submission
python transfer_preds_files.py \
--preds_pkl_dir "$PREDS_PKL" \
--preds_dir $SAVE_DIR/mov_test 

# zip the preds_dir
cd $SAVE_DIR # try to make sure no parent dir in the zip dir
zip -r mov_test.zip mov_test

# submit the prediction to the challenge server phase of Movable Prediction Test
