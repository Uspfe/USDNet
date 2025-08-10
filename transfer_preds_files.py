import os, sys, pickle
import argparse
import numpy as np
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds_pkl_dir", type=str, default=".")
    parser.add_argument("--preds_dir", type=str, default=".")
    parser.add_argument("--score_thr", type=float, default=-1)
    return parser.parse_args()

def transfer_preds_files(preds_pkl_dir, preds_dir, score_thr):
    if not os.path.exists(preds_dir):
        os.makedirs(preds_dir)
        os.makedirs(os.path.join(preds_dir, "predicted_masks"))
        os.makedirs(os.path.join(preds_dir, "predicted_articulations"))
        os.makedirs(os.path.join(preds_dir, "predicted_scores"))
        os.makedirs(os.path.join(preds_dir, "predicted_classes"))
        
    # load the preds from the preds_pkl_dir
    with open(preds_pkl_dir, 'rb') as f:
        preds = pickle.load(f)
        
    # transfer the preds to the preds_dir
    # the format of the preds in pickle is {
    #     "scene_id_1": {
    #         "pred_masks": numpy.Array, // binary mask over mesh vertices (Num_vertices, num_pred_parts)
    #         "pred_scores": numpy.Array, // confidence scores for each predicted part (num_pred_parts)
    #         "pred_classes": numpy.Array, // class labels for each predicted part (num_pred_parts), 1: rotation, 2: translation
    #         "pred_origins": numpy.Array, // axis origin for each predicted part (num_pred_parts, 3)
    #         "pred_axises": numpy.Array, // axis direction for each predicted part (num_pred_parts, 3)
    #     },
    #     ...
    #     "scene_id_2": {
    #         ...
    #     }
    #     }
    # we need to transfer the preds to the preds_dir similar to the scannet 3D instance segmentation format
    #     unzip_root/
    #  |-- scene_id_1.txt
    #  |-- scene_id_2.txt
    #  |-- scene_id_3.txt
    #      ⋮
    #  |-- predicted_masks/
    #     |-- scene_id_1_obj_id1.txt
    #     |-- scene_id_1_obj_id2.txt
    #     |-- scene_id_1_obj_id3.txt
    #     |-- ...
    #     |-- scene_id_2_obj_id1.txt
    #     |-- ...
    #  |--predicted_articulations/
    #     |-- scene_id_1_obj_id1.txt
    #     |-- scene_id_1_obj_id2.txt
    #     |-- scene_id_1_obj_id3.txt
    #     |-- ...
    #     |-- scene_id_2_obj_id1.txt
    #     |-- ...

        
        
    for scene_id, pred in tqdm.tqdm(preds.items()):
        # transform "pred_masks": numpy.Array, // binary mask over mesh vertices (Num_vertices, num_pred_parts) to a np array of shape (Num_vertices, 1)
        
        for obj_idx in range(pred["pred_masks"].shape[1]):
            obj_id = obj_idx + 1
            obj_score = pred["pred_scores"][obj_idx]
            obj_class = pred["pred_classes"][obj_idx]
            
            if 'pred_origins' in pred and 'pred_axises' in pred:
                obj_origin = pred["pred_origins"][obj_idx]
                obj_axis = pred["pred_axises"][obj_idx]
            
            if obj_score < score_thr:
                continue
            
            obj_mask = pred["pred_masks"][:, obj_idx]
            with open(os.path.join(preds_dir, scene_id + ".txt"), 'a') as f:
                mask_relative_path = os.path.join("predicted_masks", scene_id + "_" + str(obj_id) + ".txt")
                articulation_relative_path = os.path.join("predicted_articulations", scene_id + "_" + str(obj_id) + ".txt")
                f.write("{} {} {} {} {}\n".format(obj_id, mask_relative_path, articulation_relative_path, obj_score, obj_class))
                
            # save the mask
            np.savetxt(os.path.join(preds_dir, mask_relative_path), obj_mask.astype(int), fmt='%d')
            if 'pred_origins' in pred and 'pred_axises' in pred:
                # save the articulation
                np.savetxt(os.path.join(preds_dir, articulation_relative_path), np.concatenate([obj_origin, obj_axis]), fmt='%f')
                

def main():
    args = parse_args()
    preds_pkl_dir = args.preds_pkl_dir
    preds_dir = args.preds_dir
    score_thr = args.score_thr
    transfer_preds_files(preds_pkl_dir, preds_dir, score_thr)
    
if __name__ == "__main__":
    main()