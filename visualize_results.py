import open3d as o3d
from pathlib import Path
import numpy as np
import argparse

try:
    import h5py
except ImportError:
    h5py = None


def parse_prediction_rows(predicted_file: Path):
    rows = []
    with open(predicted_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "obj_id": int(parts[0]),
                    "mask_rel": parts[1],
                    "arti_rel": parts[2],
                    "score": float(parts[3]),
                    "cls": int(parts[4]),
                }
            )
    return rows


def class_color(class_id: int):
    # 1: rotation (orange), 2: translation (cyan)
    if class_id == 1:
        return np.array([1.0, 0.55, 0.1], dtype=np.float64)
    if class_id == 2:
        return np.array([0.1, 0.8, 1.0], dtype=np.float64)
    return np.array([0.8, 0.8, 0.8], dtype=np.float64)


def gt_class_color(class_id: int):
    # 1: rotation (red), 2: translation (blue)
    if class_id == 1:
        return np.array([1.0, 0.2, 0.2], dtype=np.float64)
    if class_id == 2:
        return np.array([0.2, 0.45, 1.0], dtype=np.float64)
    return np.array([0.6, 0.6, 0.6], dtype=np.float64)


def load_dict_from_h5group(h5group):
    data_dict = {}
    for key, item in h5group.items():
        try:
            int_key = int(key)
        except ValueError:
            int_key = key

        if isinstance(item, h5py.Group):
            data_dict[int_key] = load_dict_from_h5group(item)
        else:
            data_dict[int_key] = item[()]
    return data_dict


def load_gt_articulation_file(artic_file: Path):
    if h5py is None:
        raise ImportError(
            "h5py is required to load articulation ground truth files (.h5). "
            "Install it in your environment to use --draw-gt."
        )

    with h5py.File(artic_file, "r") as h5file:
        data = load_dict_from_h5group(h5file)

    gt_rows = []
    for mov_id, item in data.items():
        if not isinstance(item, dict):
            continue
        if "origin" not in item or "axis" not in item:
            continue

        sem_id = int(np.asarray(item.get("sem_id", 0)).reshape(-1)[0])
        origin = np.asarray(item["origin"], dtype=np.float64).reshape(-1)
        axis = np.asarray(item["axis"], dtype=np.float64).reshape(-1)
        if origin.shape[0] < 3 or axis.shape[0] < 3:
            continue

        gt_rows.append(
            {
                "obj_id": int(mov_id),
                "cls": sem_id,
                "origin": origin[:3],
                "axis": axis[:3],
            }
        )
    return gt_rows


def _rotation_from_z(axis_unit: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(z, axis_unit)
    c = float(np.dot(z, axis_unit))
    s = float(np.linalg.norm(v))

    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-degree rotation around x-axis maps +z to -z.
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )

    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def make_axis_arrow(origin: np.ndarray, axis: np.ndarray, color: np.ndarray, length: float):
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return None
    axis = axis / norm

    shaft_length = length * 0.75
    head_length = length * 0.25
    cylinder_radius = max(length * 0.03, 0.002)
    cone_radius = max(length * 0.08, 0.004)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=shaft_length,
        cone_height=head_length,
    )
    arrow.compute_vertex_normals()
    arrow.paint_uniform_color(color.tolist())

    # Open3D arrow points along +Z from z=0 to z=length. Rotate and place at origin.
    rotation = _rotation_from_z(axis)
    arrow.rotate(rotation, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    arrow.translate(origin.astype(np.float64))

    return arrow


def make_origin_marker(origin: np.ndarray, color: np.ndarray, radius: float):
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sph.compute_vertex_normals()
    sph.paint_uniform_color(color.tolist())
    sph.translate(origin.astype(np.float64))
    return sph

def build_point_cloud(arr: np.ndarray) -> o3d.geometry.PointCloud:
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(
            f"Expected a 2D array with at least 3 columns for XYZ, got {arr.shape}"
        )

    pcd = o3d.geometry.PointCloud()

    xyz = arr[:, :3].astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Preprocessed files store RGB in columns [3:6] as 0..255.
    if arr.shape[1] >= 6:
        rgb = arr[:, 3:6].astype(np.float64)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Normals are in columns [6:9] for this dataset.
    if arr.shape[1] >= 9:
        normals = arr[:, 6:9].astype(np.float64)
        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        nrm[nrm == 0.0] = 1.0
        normals = normals / nrm
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def main(
    scene: str,
    split: str,
    root_dir: Path,
    score_thr: float,
    topk: int,
    draw_pred: bool,
    draw_gt: bool,
):
    input_data_dir = root_dir / "data/processed/articulate3d_challenge_mov" / split
    predictions_dir = root_dir / "results/inference_mov_articulation/mov_test"
    predicted_file = predictions_dir / f"{scene}.txt"
    gt_artic_file = input_data_dir / f"{scene}_articulation.h5"

    scene_file = input_data_dir / f"{scene}.npy"
    if not scene_file.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_file}")
    if draw_pred and not predicted_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {predicted_file}")
    if draw_gt and not gt_artic_file.exists():
        raise FileNotFoundError(f"GT articulation file not found: {gt_artic_file}")

    arr = np.load(scene_file)
    print(f"Loaded: {scene_file}")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")

    pcd = build_point_cloud(arr)

    # Axis length relative to scene size.
    xyz = arr[:, :3]
    scene_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    axis_length = max(scene_diag * 0.06, 0.04)
    marker_radius = max(scene_diag * 0.004, 0.01)

    geometries = [pcd]
    pred_origins_for_stats = []
    gt_origins_for_stats = []

    if draw_pred:
        pred_rows = parse_prediction_rows(predicted_file)
        pred_rows = [r for r in pred_rows if r["score"] >= score_thr]
        pred_rows.sort(key=lambda r: r["score"], reverse=True)
        if topk > 0:
            pred_rows = pred_rows[:topk]

        pred_used = 0
        for row in pred_rows:
            arti_path = predictions_dir / row["arti_rel"]
            if not arti_path.exists():
                continue
            vals = np.loadtxt(arti_path, dtype=np.float64).reshape(-1)
            if vals.shape[0] < 6:
                continue
            origin = vals[:3]
            axis = vals[3:6]
            color = class_color(row["cls"])
            arrow = make_axis_arrow(origin, axis, color, axis_length)
            if arrow is not None:
                geometries.append(arrow)
                geometries.append(make_origin_marker(origin, color, marker_radius))
                pred_origins_for_stats.append(origin)
                pred_used += 1
        print(f"Loaded predictions: {len(pred_rows)}, visualized articulations: {pred_used}")

    if draw_gt:
        gt_rows = load_gt_articulation_file(gt_artic_file)
        gt_used = 0
        for row in gt_rows:
            origin = row["origin"]
            axis = row["axis"]
            color = gt_class_color(row["cls"])
            arrow = make_axis_arrow(origin, axis, color, axis_length * 0.9)
            if arrow is not None:
                geometries.append(arrow)
                geometries.append(make_origin_marker(origin, color, marker_radius * 1.1))
                gt_origins_for_stats.append(origin)
                gt_used += 1
        print(f"Loaded GT articulations: {len(gt_rows)}, visualized articulations: {gt_used}")

    if draw_pred:
        print("Pred color legend: rotation=orange (1), translation=cyan (2)")
    if draw_gt:
        print("GT color legend: rotation=red (1), translation=blue (2)")

    if pred_origins_for_stats:
        origins = np.vstack(pred_origins_for_stats)
        nearest = np.sqrt(((xyz[None, :, :] - origins[:, None, :]) ** 2).sum(axis=2)).min(axis=1)
        print(
            "Pred origin alignment (nearest-point dist) mean="
            f"{nearest.mean():.4f}, median={np.median(nearest):.4f}, max={nearest.max():.4f}"
        )

    if gt_origins_for_stats:
        origins = np.vstack(gt_origins_for_stats)
        nearest = np.sqrt(((xyz[None, :, :] - origins[:, None, :]) ** 2).sum(axis=2)).min(axis=1)
        print(
            "GT origin alignment (nearest-point dist) mean="
            f"{nearest.mean():.4f}, median={np.median(nearest):.4f}, max={nearest.max():.4f}"
        )

    o3d.visualization.draw_geometries(geometries, window_name=f"{scene} ({split})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Articulate3D point clouds")
    parser.add_argument("--scene", default="0a76e06478", help="Scene id without .npy")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--root", default=".", help="USDNet repo root")
    parser.add_argument("--score-thr", type=float, default=0.2, help="Minimum prediction score")
    parser.add_argument("--topk", type=int, default=30, help="Top-K predictions to draw after threshold; <=0 means all")
    parser.add_argument(
        "--draw-pred",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Draw model predictions (default: true)",
    )
    parser.add_argument(
        "--draw-gt",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Draw GT articulations from <scene>_articulation.h5 (default: true)",
    )
    args = parser.parse_args()

    main(
        args.scene,
        args.split,
        Path(args.root).resolve(),
        args.score_thr,
        args.topk,
        args.draw_pred,
        args.draw_gt,
    )