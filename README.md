# Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description
This repository contains the official code release for the **Articulate3D** paper, accepted at **ICCV 2025**. It provides the **USDNet baseline implementation** as well as the **SceneDataLoader** for the Articulate3D dataset.  

> 📄 **Paper**: [Articulate3D (ICCV 2025)](https://insait-institute.github.io/articulate3d.github.io/)  
> 🏁 **Challenge**: Track 3 at [OpenSUN3D Workshop, ICCV 2025](https://opensun3d.github.io/)  
> 🤗 **Dataset**: Articulate3D is available on [HuggingFace](https://huggingface.co/datasets/INSAIT-Institute/Articulate3D)

---


## 📦 What's in this repo?

Currently released:
- [USDNet](#usdnet): Implementation of the baseline for Articulate3D challenge tasks.  
- [SceneDataLoader](#-scenedataloader-documentation-scenedataloaderpy): Python class for loading and parsing Articulate3D annotations.  

## 🚀 Challenge Participation

Join the **Articulate3D Challenge** at the **OpenSUN3D Workshop (ICCV 2025)**!  
We're hosting **Track 3**, which focuses on articulated scene understanding.

📍 Challenge details and submission portal: [OpenSUN3D Challenge](https://opensun3d.github.io/)

---
## USDNet

### 1. Code structure
We adapt the codebase of [Mask3D](https://github.com/JonasSchult/Mask3D) which provides a highly modularized framework for 3D Semantic Instance Segmentation based on the MinkowskiEngine.

```
├── USDNet
│   ├── main_instance_segmentation_articulation.py <- the main file
│   ├── conf                          <- hydra configuration files
│   ├── datasets
│   │   ├── preprocessing             <- folder with preprocessing scripts
│   │   │   ├── articulate3d_preprocessing_challenge.py   <- file of preprocessing for the challenge
│   │   ├── semseg.py                 <- indoor dataset
│   │   └── utils.py        
│   ├── models                        <- USDNet model based on Mask3D
│   ├── trainer
│   │   ├── __init__.py
│   │   └── trainer.py                <- train loop
│   └── utils
├── data
│   ├── processed                     <- folder for preprocessed datasets
│   └── raw                           <- folder for raw datasets
├── scripts                           <- train scripts
├── docs
├── README.md
└── saved                             <- folder that stores models and logs
└──Dockerfile                         <- Dockerfile for env setup for cuda 12
```

### 2. Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.10.9
cuda: 11.3
```
You can set up a conda environment following instructions in [Mask3D](https://github.com/JonasSchult/Mask3D).

We also provide a Docker file (./Dockerfile) for the environment setup for cuda: 12.1:
```
docker build -t usdnet:latest .
```

### 3. Data preprocessing :hammer:
After installing the dependencies, we preprocess the datasets. 
Note we also provide the preprocessed data [here](https://drive.google.com/drive/folders/1HYPkUnF5QIdV2gH9vwgWW4lAvCSg51jK?usp=drive_link) for the convenience. You can download it and put it in the ./data/processed and skip the following preprocessing steps.

First, put the dataset in the dir "./data/raw/articulate3d".
Then run the [bash file](./scripts/preprocessing_articulate3d.sh) and the preprocessed files will be saved in "./data/processed/".
For efficiency, the preprocessing code will downsample the pointcloud of the mesh from [Scannet++](https://kaldir.vc.in.tum.de/scannetpp/) with voxel size 0.01 cm. 
Note that the evaluation in [Articulate3D challenge](https://insait-institute.github.io/articulate3d.github.io/challenge.html) is based on the voxelized point cloud with the ground truth annotations.

Note the splits files is in "./datasets/articulate3d" and should be copied to "./data/raw/articulate3d/".

The structure should look like this:
```
├── USDNet
│   ├── data
│   │   ├── raw                       <- raw data
│   │   │   ├── articulate3d
│   │   │   │   ├──splits             <- splits of training, validation and test set
│   │   │   │   │   ├──train.txt
│   │   │   │   │   ├──val.txt
│   │   │   │   │   ├──tesst.txt
│   │   │   │   │──scans
│   │   │   │   │   ├──0a5c013435
│   │   │   │   │   │   ├──mesh_aligned_0.05.ply      <- mesh file
│   │   │   │   │   │   ├──0a5c013435_parts.json      <- annotation for movable and interactable part segmentation
│   │   │   │   │   │   ├──0a5c013435_artic.json      <- annotation for articulation parameters of movable part
│   │   │   │   │   ├── ... 
│   │   ├── processed                 <- folder with processed data by preprocessing_articulate3d.sh 
│   │   │   ├──articulate3d_challenge_mov             <- processed data for movable part seg and articulation prediction
│   │   │   │   │──train                              <- dataset with pointcloud, color and normal  + annotation for training set 
│   │   │   │   │──validation                         <- dataset with pointcloud, color and normal  + annotation for validation set 
│   │   │   │   │──test                               <- dataset with pointcloud, color and normal  + annotation for test set 
│   │   │   │   │──train_database.yaml                <- database for train set, used for dataloader to locate file paths
│   │   │   │   │──validation_database.yaml           <- database for validation set
│   │   │   │   │──train_validation_database.yaml     <- database for train+validation set
│   │   │   │   │──test_database.yaml                 <- database for test set
│   │   │   │   │──expand_dict                        <- neighbored point annotation of movable part, for coarse to fine segmentation training
│   │   │   │   │──instance_gt                        <- gt segmentation annotation in .txt

```

### 4. Training :train2:
### Movable part segmentation and articulation prediction
Step 1

Download the [pretrained model](https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_benchmark.ckpt) of Mask3D. 

Step 2

Check the notes and TODOs in the "./scripts/train_mov.sh" to set the correct key and path

Step 3

Start training for movable part segmentation and articulation parameter prediction:
```bash
bash ./scripts/train_mov.sh
```

#### Interactable part segmentation
Step 1 

Get the trained model from "Movable part segmentation and articulation prediction" and use it for training interactable part segmentation to speed up converging. 

Step 2

Check the notes and TODOs in the "./scripts/inter_mov.sh" to set the correct key and path

In the simplest case the inference command looks as follows:

Step 3

Start training for interactable part segmentation:
```bash
bash ./scripts/train_inter.sh
```

#### Trained checkpoints :floppy_disk:
We provide the trained checkpoints for the 2 tasks [here](https://drive.google.com/drive/folders/1qv2hTF8_U7nM1tAz1hltO1EGggstVaeR?usp=sharing).


### 5. Inference :chart_with_upwards_trend:
Run inference script for evaluation of the trained mode and for the challange submission

#### Movable part segmentation and articulation prediction
```bash
bash ./scripts/infer_mov.sh
```

#### Interactable part segmentation
```bash
bash ./scripts/infer_inter.sh
```

## 📂 SceneDataLoader Documentation: `SceneDataLoader.py`

### Overview

`SceneDataLoader` is a Python iterator that loads Articulate3D annotations from its dataset directory.  
Each scene is composed of:
- `<scene_id>_parts.json`: part annotations and mesh face indices.
- `<scene_id>_artic.json`: articulation parameters (axis, origin, range, type).

### Usage

```python
from loader import SceneDataLoader

loader = SceneDataLoader("path/to/Articulate3D/")
for scene_id, scene_dict, face_mask in loader:
    print(f"Scene: {scene_id}")
    print(f"Articulated parts: {list(scene_dict.keys())}")
    print(f"Face mask shape: {face_mask.shape}")
```


## TODO List
- [x] Release Code
- [x] Set up Challenge Server
- [x] Training Code and instructions
- [x] Checkpoints (mov yes, inter no)
- [x] provide preprocessed data for user's convenience
- [x] Add docker file for env setup in cuda 12
- [ ] Merge data loader with json format to datapreprocessing


## BibTeX :pray:
```
@InProceedings{halacheva2024articulate3d,
    author    = {Halacheva, Anna-Maria and Miao, Yang and Zaech, Jan-Nico and Wang, Xi and Van Gool, Luc and Paudel, Danda Pani},
    title     = {Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2025},
  }
```
