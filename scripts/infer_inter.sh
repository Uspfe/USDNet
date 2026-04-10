#!/bin/bash

# inference the interactable part segmentation  

# NOTE:
# 1. Please check TODOs before running the script
# 2. general.debug is set to true to save the predictions in pickle format

cd USDNet # TODO: change the path to the USDNet directory

export WANDB_MODE="offline"
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_LAUNCH_BLOCKING=1
GPUS=1
CURR_DBSCAN=0.95
CURR_TOPK=150
CURR_QUERY=100

BACKBONE_CKPT="./checkpoints/inter_trainval.ckpt"
SAVE_DIR=./results/inference_inter_seg

python main_instance_segmentation_articulation.py \
general.experiment_name="inference_inter_seg" \
general.project_name="articulate3d_challenge" \
data/datasets=articulate3d_challenge_inter \
general.num_targets=4 \
general.eval_on_segments=false \
general.train_on_segments=false \
general.save_dir=$SAVE_DIR \
general.eval_articulation=false \
general.eval_hierarchy_inter=false \
general.debug=true \
general.train_mode=false \
data.train_mode="train_validation" \
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
data.test_mode="validation" \
model.num_queries=${CURR_QUERY} \
model.predict_articulation_mode=2 \
model.predict_hierarchy_interaction=false \
model.predict_articulation=false \
loss.regular_arti_loss=false \
loss.losses="[labels,masks]" \
trainer.check_val_every_n_epoch=20 \
optimizer.lr=0.0001 \
general.checkpoint=$BACKBONE_CKPT

# transfer the prediction from .pickle to the format of the scannet 3D instance segmentation in .zip to save space for submission
python transfer_preds_files.py \
--preds_pkl_dir $SAVE_DIR/debug/val_preds/preds.pkl \
--preds_dir $SAVE_DIR/inter_test 

# zip the preds_dir
cd $SAVE_DIR # try to make sure no parent dir in the zip dir
zip -r inter_test.zip inter_test

# submit the prediction to the challenge server phase of Interactable Prediction Test
