#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one custom NPZ (xyz/rgb) into USDNet processed dataset format"
    )
    parser.add_argument("--input_npz", required=True, help="Path to custom .npz file")
    parser.add_argument(
        "--output_root",
        default="data/processed/custom_mov_npz",
        help="Output processed dataset root",
    )
    parser.add_argument(
        "--scene_id",
        default="",
        help="Optional scene id. Default: NPZ filename stem",
    )
    parser.add_argument(
        "--segment_mode",
        choices=["single", "per_point"],
        default="per_point",
        help="How to fill segment ids for inference-only data",
    )
    return parser.parse_args()


def _load_xyz_rgb(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    if "xyz" not in data or "rgb" not in data:
        raise ValueError("Input NPZ must contain 'xyz' and 'rgb' arrays")

    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape [N, 3], got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb shape [N, 3], got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must contain the same number of points")

    if rgb.max() <= 1.0:
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
    else:
        rgb = np.clip(rgb, 0.0, 255.0)

    return xyz, rgb


def _build_scene_array(
    xyz: np.ndarray,
    rgb: np.ndarray,
    segment_mode: str,
) -> np.ndarray:
    num_points = xyz.shape[0]

    normals = np.zeros((num_points, 3), dtype=np.float32)
    sem_labels = np.ones((num_points, 1), dtype=np.float32)
    instance_ids = np.ones((num_points, 1), dtype=np.float32)

    if segment_mode == "single":
        segment_ids = np.zeros((num_points, 1), dtype=np.float32)
    else:
        segment_ids = np.arange(num_points, dtype=np.float32).reshape(-1, 1)

    interaction_ids = np.zeros((num_points, 1), dtype=np.float32)

    return np.concatenate(
        [
            xyz.astype(np.float32),
            rgb.astype(np.float32),
            normals,
            sem_labels,
            instance_ids,
            segment_ids,
            interaction_ids,
        ],
        axis=1,
    )


def _write_yaml(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _color_stats(rgb: np.ndarray) -> dict:
    rgb01 = np.clip(rgb / 255.0, 0.0, 1.0)
    mean = rgb01.mean(axis=0).astype(float).tolist()
    std = rgb01.std(axis=0)
    std = np.maximum(std, 1e-6)
    return {"mean": mean, "std": std.astype(float).tolist()}


def main() -> None:
    args = parse_args()
    input_npz = Path(args.input_npz)
    if not input_npz.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_npz}")

    output_root = Path(args.output_root)
    scene_id = args.scene_id.strip() or input_npz.stem

    xyz, rgb = _load_xyz_rgb(input_npz)
    scene_arr = _build_scene_array(xyz, rgb, args.segment_mode)

    test_dir = output_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)

    scene_npy = test_dir / f"{scene_id}.npy"
    np.save(scene_npy, scene_arr)

    db_entry = {
        "filepath": str(scene_npy.as_posix()),
        "raw_filepath": f"custom_npz/{scene_id}.npy",
        "scene": scene_id,
        "instance_gt_filepath": [],
    }
    db_entries = [db_entry]

    _write_yaml(output_root / "test_database.yaml", db_entries)
    _write_yaml(output_root / "validation_database.yaml", db_entries)
    _write_yaml(output_root / "train_validation_database.yaml", db_entries)

    label_db = {
        0: {"name": "background", "validation": True},
        1: {"name": "rotation", "validation": True},
        2: {"name": "translation", "validation": True},
    }
    _write_yaml(output_root / "label_database.yaml", label_db)
    _write_yaml(output_root / "color_mean_std.yaml", _color_stats(rgb))

    print(f"Prepared custom scene: {scene_id}")
    print(f"Saved point cloud: {scene_npy}")
    print(f"Saved metadata root: {output_root}")


if __name__ == "__main__":
    main()
