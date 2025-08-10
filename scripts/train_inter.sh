#!/bin/bash

# training the interactable part segmentation

# NOTE:
# 1. Please check TODOs before running the script
# 2. If the OOM error due to the limited gpu memory, set data.cropping=true to crop the input point cloud during training, which, meanwhile, can lead to decreased performance 
# 3. the code base is based on Mask3D, in which multi-gpu training is not supported. Please use single-gpu training for now.


cd USDNet # TODO: change the path to the USDNet directory

export WANDB_API_KEY="your wandb api key for log" # TODO: set the API key of your WANDB account
export WANDB_MODE="online"
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_LAUNCH_BLOCKING=1
GPUS=1
CURR_DBSCAN=0.95
CURR_TOPK=150
CURR_QUERY=100

# TRAIN
BACKBONE_CKPT= # TODO: set the path of Mask3D pre-trained model
TRAIN_MODE="train" # "train" means training with train set for dev mode and "train_validation" means training with train+validation sets for submission in test set
SAVE_DIR=./results/train_inter_seg 

python main_instance_segmentation_articulation.py \
general.experiment_name="train_inter_seg" \
general.project_name="articulate3d_challenge" \
data/datasets=articulate3d_challenge_inter \
general.num_targets=4 \
general.eval_on_segments=false \
general.train_on_segments=false \
general.save_dir=$SAVE_DIR \
general.eval_articulation=false \
general.eval_hierarchy_inter=false \
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
model.num_queries=${CURR_QUERY} \
model.predict_articulation_mode=2 \
model.predict_hierarchy_interaction=false \
model.predict_articulation=false \
loss.regular_arti_loss=false \
loss.losses="[labels,masks]" \
trainer.check_val_every_n_epoch=20 \
optimizer.lr=0.0001 \
general.checkpoint=$BACKBONE_CKPT
