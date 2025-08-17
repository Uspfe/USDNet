#!/bin/bash

cd USDNet

# movable part data
python -m datasets.preprocessing.articulate3d_preprocessing_challenge preprocess \
--data_dir="./data/raw/articulate3d" \
--save_dir="./data/processed/articulate3d_challenge_mov" \
--exclude_stuff=True \
--gt_annos_testset=False \
--n_jobs=8

# interactable part
python -m datasets.preprocessing.articulate3d_preprocessing_challenge preprocess \
--data_dir="./data/raw/articulate3d" \
--save_dir="./data/processed/articulate3d_challenge_inter" \
--exclude_stuff=True \
--gt_annos_testset=False \
--interaction_as_movement=True \
--n_jobs=8