import logging
from pathlib import Path
import json

import hydra
import numpy as np
import open3d as o3d
from omegaconf import DictConfig
import random


logger = logging.getLogger(__name__)


def process_scene(
    artic_gt_scene_path: Path,
    parts_gt_scene_path: Path,
    cropped_scene_path: Path,
    output_dir: Path,
    min_visibility_threshold: float,
    structured_layout: bool = True,
    scannet_root: Path | None = None,
) -> None:
    if not artic_gt_scene_path.exists():
        logger.warning(
            f"Articulation ground truth file not found: {artic_gt_scene_path}"
        )
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

    # Build sorted index lists so the cropped mesh has a deterministic vertex/triangle order
    visible_vert_list = sorted(view["visible_mesh_indices"])
    visible_tri_list = sorted(view["visible_triangle_indices"])
    visible_vert_set = set(visible_vert_list)
    visible_tri_set = set(visible_tri_list)

    # Remapping dicts: original index -> index in cropped mesh
    old_vert_to_new = {old: new for new, old in enumerate(visible_vert_list)}
    old_tri_to_new = {old: new for new, old in enumerate(visible_tri_list)}

    # --- crop part annotations and remap indices to cropped mesh ---
    cropped_annotations = []
    visible_part_ids = set()

    for ann in parts_gt["data"]["annotations"]:
        vis_verts = [v for v in ann["vertIndices"] if v in visible_vert_set]
        if not vis_verts:
            continue
        visibility_ratio = len(vis_verts) / len(ann["vertIndices"])
        visible_part_ids.add(ann["partId"])
        vis_tris = [t for t in ann["triIndices"] if t in visible_tri_set]
        cropped_ann = dict(ann)
        cropped_ann["vertIndices"] = [old_vert_to_new[v] for v in vis_verts]
        cropped_ann["triIndices"] = [old_tri_to_new[t] for t in vis_tris]
        cropped_ann["visibilityRatio"] = visibility_ratio
        cropped_annotations.append(cropped_ann)

    if not visible_part_ids:
        logger.warning(f"No visible parts for {cropped_scene_path.name}")

    # recompute stats against the cropped mesh dimensions
    n_cropped_verts = len(visible_vert_list)
    n_cropped_faces = len(visible_tri_list)
    ann_faces = sum(len(a["triIndices"]) for a in cropped_annotations)
    orig_stats = parts_gt["data"]["stats"]
    cropped_stats = {
        "annotatedFaces": ann_faces,
        "unannotatedFaces": n_cropped_faces - ann_faces,
        "totalFaces": n_cropped_faces,
        "annotatedFaceArea": 0,
        "unannotatedFaceArea": orig_stats["unannotatedFaceArea"],
        "totalFaceArea": orig_stats["totalFaceArea"],
        "percentComplete": ann_faces / n_cropped_faces * 100 if n_cropped_faces else 0,
        "totalVertices": n_cropped_verts,
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
            "parts": [
                p for p in artic_gt["data"]["parts"] if p["pid"] in visible_part_ids
            ],
            "articulations": [
                a
                for a in artic_gt["data"]["articulations"]
                if (
                    a["pid"] in visible_part_ids
                    and all(base_pid in visible_part_ids for base_pid in a["base"])
                )
            ],
        },
    }

    # --- prepare output directory ---
    stem = cropped_scene_path.stem
    if structured_layout:
        save_dir = output_dir / "scans" / stem
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

    # --- crop and save mesh ---
    scene_id = view["scene_id"]
    mesh_out = save_dir / "mesh_aligned_0.05.ply"
    if not mesh_out.exists():
        if scannet_root is not None:
            mesh_src = scannet_root / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
        else:
            mesh_src = None
        if mesh_src is not None and mesh_src.exists():
            mesh = o3d.io.read_triangle_mesh(str(mesh_src))
            all_verts = np.asarray(mesh.vertices)
            all_tris = np.asarray(mesh.triangles)
            new_verts = all_verts[visible_vert_list]
            # remap triangle vertex indices to cropped mesh
            new_tris = np.vectorize(old_vert_to_new.get)(all_tris[visible_tri_list])
            cropped_mesh = o3d.geometry.TriangleMesh()
            cropped_mesh.vertices = o3d.utility.Vector3dVector(new_verts)
            cropped_mesh.triangles = o3d.utility.Vector3iVector(new_tris)
            if mesh.has_vertex_colors():
                cropped_mesh.vertex_colors = o3d.utility.Vector3dVector(
                    np.asarray(mesh.vertex_colors)[visible_vert_list]
                )
            if mesh.has_vertex_normals():
                cropped_mesh.vertex_normals = o3d.utility.Vector3dVector(
                    np.asarray(mesh.vertex_normals)[visible_vert_list]
                )
            o3d.io.write_triangle_mesh(str(mesh_out), cropped_mesh)
        else:
            logger.warning(f"Mesh not found, skipping mesh crop: {mesh_src}")

    with open(save_dir / f"{stem}_artic.json", "w") as f:
        json.dump(cropped_artic, f, indent=2)
    with open(save_dir / f"{stem}_parts.json", "w") as f:
        json.dump(cropped_parts, f, indent=2)


def create_splits(
    cropped_data_path: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    test_ratio: float = 0.15,
    artic_ratio: float | None = None,
    structured_layout: bool = True,
    seed: int = 42,
) -> None:
    """Write train.txt / val.txt / test.txt with stratified splits.

    Each split gets the same ratio of has-articulation to no-articulation
    views. If artic_ratio is given (0-1), the larger group is subsampled so
    that exactly that fraction of views has articulations; otherwise the
    natural ratio is preserved.
    """

    dataset = json.load(open(cropped_data_path, "r"))

    has_artic: list[str] = []
    no_artic: list[str] = []

    for view_id in dataset:
        artic_path = (
            output_dir / "scans" / view_id / f"{view_id}_artic.json"
            if structured_layout
            else output_dir / f"{view_id}_artic.json"
        )
        if not artic_path.exists():
            logger.warning(f"Annotation file not found, skipping: {artic_path.name}")
            continue
        artic = json.load(open(artic_path))
        if artic["data"]["articulations"]:
            has_artic.append(view_id)
        else:
            no_artic.append(view_id)

    rng = random.Random(seed)
    rng.shuffle(has_artic)
    rng.shuffle(no_artic)

    if artic_ratio is not None:
        if not (0 < artic_ratio < 1):
            raise ValueError(f"artic_ratio must be between 0 and 1, got {artic_ratio}")
        # subsample the larger group to hit the requested ratio
        n_total = min(len(has_artic) / artic_ratio, len(no_artic) / (1 - artic_ratio))
        has_artic = has_artic[: int(n_total * artic_ratio)]
        no_artic = no_artic[: int(n_total * (1 - artic_ratio))]

    logger.info(
        f"Split pool: {len(has_artic)} with articulations, {len(no_artic)} without "
        f"(artic fraction: {len(has_artic) / (len(has_artic) + len(no_artic)):.2f})"
    )

    def split_group(ids: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(ids)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)
        return ids[:n_train], ids[n_train : n_train + n_test], ids[n_train + n_test :]

    train_h, test_h, val_h = split_group(has_artic)
    train_n, test_n, val_n = split_group(no_artic)

    has_artic_set = set(has_artic)
    splits = {
        "train": train_h + train_n,
        "test": test_h + test_n,
        "val": val_h + val_n,
    }

    splits_dir = output_dir / "splits" if structured_layout else output_dir
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in splits.items():
        rng.shuffle(ids)
        out_path = splits_dir / f"{split_name}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(ids) + "\n")
        n_with = sum(i in has_artic_set for i in ids)
        logger.info(
            f"{split_name}: {len(ids)} views ({n_with} with articulations, {len(ids) - n_with} without)"
        )


@hydra.main(
    version_base="1.1",
    config_path="./config",
    config_name="config_data_crop_groundtruth",
)
def crop_groundtruth(cfg: DictConfig) -> None:
    import os

    articulate3d_gt_path = Path(cfg.crop_groundtruth.articulate3d_dir).absolute()
    cropped_data_path = Path(cfg.crop_groundtruth.cropped_scenes).absolute()
    output_dir = Path(cfg.crop_groundtruth.output_dir).absolute()
    structured_layout: bool = cfg.crop_groundtruth.structured_layout

    scannet_root_cfg = cfg.crop_groundtruth.scannet_root
    if scannet_root_cfg is not None:
        scannet_root = Path(scannet_root_cfg).absolute()
    elif "SCANNET_ROOT" in os.environ:
        scannet_root = Path(os.environ["SCANNET_ROOT"])
    else:
        scannet_root = None
        if structured_layout:
            logger.warning("SCANNET_ROOT not set and scannet_root not configured — mesh symlinks will be skipped")

    cropped_scenes = json.load(open(cropped_data_path, "r"))
    for id, scene_dict in cropped_scenes.items():
        scannet_scene = scene_dict["scene_id"]
        artic_gt_scene_path = articulate3d_gt_path / f"{scannet_scene}_artic.json"
        parts_gt_scene_path = articulate3d_gt_path / f"{scannet_scene}_parts.json"
        cropped_scene_path = cropped_data_path.parent / scene_dict["path"]
        process_scene(
            artic_gt_scene_path,
            parts_gt_scene_path,
            cropped_scene_path,
            output_dir,
            cfg.crop_groundtruth.min_visibility_threshold,
            structured_layout=structured_layout,
            scannet_root=scannet_root,
        )

    create_splits(
        cropped_data_path,
        output_dir,
        train_ratio=cfg.crop_groundtruth.split.train_ratio,
        test_ratio=cfg.crop_groundtruth.split.test_ratio,
        artic_ratio=cfg.crop_groundtruth.split.artic_ratio,
        structured_layout=structured_layout,
        seed=cfg.general.seed,
    )


if __name__ == "__main__":
    crop_groundtruth()
