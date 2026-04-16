import logging
from pathlib import Path
import json

import hydra
from omegaconf import DictConfig


logger = logging.getLogger(__name__)

def process_scene(artic_gt_scene_path: Path, parts_gt_scene_path: Path, cropped_scene_path: Path) -> None:
    if not artic_gt_scene_path.exists():
        logger.warning(f"Articulation ground truth file not found: {artic_gt_scene_path}")
        return

    if not parts_gt_scene_path.exists():
        logger.warning(f"Parts ground truth file not found: {parts_gt_scene_path}")
        return

    if not cropped_scene_path.exists():
        logger.warning(f"Cropped scene file not found: {cropped_scene_path}")
        return
    
    print("all exist")


@hydra.main(
    version_base="1.1",
    config_path="./config",
    config_name="config_data_crop_groundtruth",
)
def crop_groundtruth(cfg: DictConfig) -> None:
    articulate3d_gt_path = Path(cfg.crop_groundtruth.articulate3d_dir).absolute()
    cropped_data_path = Path(cfg.crop_groundtruth.cropped_scenes).absolute()

    cropped_scenes = json.load(open(cropped_data_path, "r"))
    for id, scene_dict in cropped_scenes.items():
        scannet_scene = scene_dict["scene_id"]
        artic_gt_scene_path = articulate3d_gt_path / f"{scannet_scene}_artic.json"
        parts_gt_scene_path = articulate3d_gt_path / f"{scannet_scene}_parts.json"
        cropped_scene_path = cropped_data_path.parent / scene_dict["path"]
        process_scene(artic_gt_scene_path, parts_gt_scene_path, cropped_scene_path)


if __name__ == "__main__":
    crop_groundtruth()