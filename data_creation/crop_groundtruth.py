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

    artic_gt = json.load(open(artic_gt_scene_path))
    parts_gt = json.load(open(parts_gt_scene_path))
    view = json.load(open(cropped_scene_path))

    visible_vert_set = set(view["visible_mesh_indices"])
    visible_tri_set = set(view["visible_triangle_indices"])

    # --- crop part annotations ---
    cropped_annotations = []
    visible_part_ids = set()

    for ann in parts_gt["data"]["annotations"]:
        vis_verts = [v for v in ann["vertIndices"] if v in visible_vert_set]
        if not vis_verts:
            continue
        visible_part_ids.add(ann["partId"])
        vis_tris = [t for t in ann["triIndices"] if t in visible_tri_set]
        cropped_ann = dict(ann)
        cropped_ann["vertIndices"] = vis_verts
        cropped_ann["triIndices"] = vis_tris
        cropped_annotations.append(cropped_ann)

    if not visible_part_ids:
        logger.warning(f"No visible parts for {cropped_scene_path.name}")

    # recompute stats for the visible subset
    ann_faces = sum(len(a["triIndices"]) for a in cropped_annotations)
    orig_stats = parts_gt["data"]["stats"]
    total_faces = orig_stats["totalFaces"]
    cropped_stats = {
        "annotatedFaces": ann_faces,
        "unannotatedFaces": total_faces - ann_faces,
        "totalFaces": total_faces,
        "annotatedFaceArea": 0,
        "unannotatedFaceArea": orig_stats["unannotatedFaceArea"],
        "totalFaceArea": orig_stats["totalFaceArea"],
        "percentComplete": ann_faces / total_faces * 100 if total_faces else 0,
        "totalVertices": orig_stats["totalVertices"],
    }

    cropped_parts = {
        "modelId": parts_gt["modelId"],
        "labels": [a["label"] for a in cropped_annotations],
        "data": {
            "stats": cropped_stats,
            "annotations": cropped_annotations,
        },
    }

    # --- crop articulation annotations ---
    cropped_artic = {
        "modelId": artic_gt["modelId"],
        "data": {
            "parts": [p for p in artic_gt["data"]["parts"] if p["pid"] in visible_part_ids],
            "articulations": [a for a in artic_gt["data"]["articulations"] if a["pid"] in visible_part_ids],
        },
    }

    # --- save alongside the view file ---
    stem = cropped_scene_path.stem
    out_dir = cropped_scene_path.parent
    with open(out_dir / f"{stem}_artic.json", "w") as f:
        json.dump(cropped_artic, f, indent=2)
    with open(out_dir / f"{stem}_parts.json", "w") as f:
        json.dump(cropped_parts, f, indent=2)


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