import open3d as o3d
from pathlib import Path
import numpy as np
import argparse


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


def make_axis_line(origin: np.ndarray, axis: np.ndarray, color: np.ndarray, length: float):
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return None
    axis = axis / norm
    start = origin - axis * (0.5 * length)
    end = origin + axis * (0.5 * length)

    points = np.vstack([start, end]).astype(np.float64)
    lines = np.array([[0, 1]], dtype=np.int32)
    colors = np.array([color], dtype=np.float64)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


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


def main(scene: str, split: str, root_dir: Path, score_thr: float, topk: int):
    input_data_dir = root_dir / "data/processed/articulate3d_challenge_mov" / split
    predictions_dir = root_dir / "results/inference_mov_articulation/mov_test"
    predicted_file = predictions_dir / f"{scene}.txt"

    scene_file = input_data_dir / f"{scene}.npy"
    if not scene_file.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_file}")
    if not predicted_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {predicted_file}")

    arr = np.load(scene_file)
    print(f"Loaded: {scene_file}")
    print(f"Shape: {arr.shape}, dtype: {arr.dtype}")

    pcd = build_point_cloud(arr)

    # Axis length relative to scene size.
    xyz = arr[:, :3]
    scene_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    axis_length = max(scene_diag * 0.06, 0.04)
    marker_radius = max(scene_diag * 0.004, 0.01)

    pred_rows = parse_prediction_rows(predicted_file)
    pred_rows = [r for r in pred_rows if r["score"] >= score_thr]
    pred_rows.sort(key=lambda r: r["score"], reverse=True)
    if topk > 0:
        pred_rows = pred_rows[:topk]

    geometries = [pcd]
    origins_for_stats = []
    used = 0
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
        line = make_axis_line(origin, axis, color, axis_length)
        if line is not None:
            geometries.append(line)
            geometries.append(make_origin_marker(origin, color, marker_radius))
            origins_for_stats.append(origin)
            used += 1

    print(f"Loaded predictions: {len(pred_rows)}, visualized articulations: {used}")
    print("Class color legend: rotation=orange (1), translation=cyan (2)")
    if origins_for_stats:
        origins = np.vstack(origins_for_stats)
        # Quick sanity check: how far articulation origins are from the nearest scene point.
        nearest = np.sqrt(((xyz[None, :, :] - origins[:, None, :]) ** 2).sum(axis=2)).min(axis=1)
        print(
            "Origin alignment (nearest-point dist) mean="
            f"{nearest.mean():.4f}, median={np.median(nearest):.4f}, max={nearest.max():.4f}"
        )
    o3d.visualization.draw_geometries(geometries, window_name=f"{scene} ({split})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Articulate3D point clouds")
    parser.add_argument("--scene", default="1a130d092a", help="Scene id without .npy")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--root", default=".", help="USDNet repo root")
    parser.add_argument("--score-thr", type=float, default=0.2, help="Minimum prediction score")
    parser.add_argument("--topk", type=int, default=30, help="Top-K predictions to draw after threshold; <=0 means all")
    args = parser.parse_args()

    main(args.scene, args.split, Path(args.root).resolve(), args.score_thr, args.topk)