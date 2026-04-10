import open3d as o3d
from pathlib import Path
import numpy as np
import argparse
import yaml

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


def load_predicted_masks_for_scene(
    predictions_dir: Path, pred_rows: list
) -> dict:
    """Load predicted masks using mask_rel paths from prediction rows.
    Returns dict: obj_id -> mask array (binary 0/1, shape (num_points,)).
    """
    masks = {}
    for row in pred_rows:
        obj_id = row["obj_id"]
        mask_rel = row["mask_rel"]
        mask_path = predictions_dir / mask_rel
        if mask_path.exists():
            try:
                mask_data = np.loadtxt(mask_path, dtype=np.int32).reshape(-1)
                masks[obj_id] = mask_data.astype(bool)
                print(f"  Loaded mask for obj {obj_id}: {mask_path} (shape={mask_data.shape})")
            except Exception as e:
                print(f"  Warning: Could not load mask {mask_path}: {e}")
        else:
            print(f"  Warning: Mask file not found: {mask_path}")
    return masks


def load_gt_masks(instance_gt_file: Path) -> np.ndarray:
    """Load GT instance labels. Returns array of shape (num_points,) with instance IDs."""
    if instance_gt_file is None or not instance_gt_file.exists():
        return None
    try:
        labels = np.loadtxt(instance_gt_file, dtype=np.int32).reshape(-1)
        print(f"  Loaded GT masks from: {instance_gt_file}")
        return labels
    except Exception as e:
        print(f"  Warning: Could not load GT masks from {instance_gt_file}: {e}")
        return None


def load_database_yaml(db_path: Path, scene: str) -> dict:
    """Load a scene entry from database YAML. Returns dict with scene metadata or None."""
    if not db_path.exists():
        print(f"  Database YAML not found: {db_path}")
        return None
    try:
        with open(db_path, "r") as f:
            database = yaml.safe_load(f)
        if not isinstance(database, list):
            return None
        for entry in database:
            if entry.get("scene") == scene:
                return entry
        print(f"  Scene {scene} not found in database")
        return None
    except Exception as e:
        print(f"  Warning: Could not load database YAML from {db_path}: {e}")
        return None


def generate_instance_colors(num_instances: int) -> dict:
    """Generate distinct colors for instances using HSV space.
    Returns dict: instance_id -> RGB array.
    """
    colors = {}
    if num_instances == 0:
        return colors
    for i in range(num_instances):
        hue = (i * 0.618) % 1.0  # Golden ratio for perceptual spacing
        sat = 0.7 + (i % 3) * 0.1  # Vary saturation
        val = 0.75 + (i % 2) * 0.15  # Vary brightness
        rgb = np.array(
            _hsv_to_rgb(hue, sat, val),
            dtype=np.float64,
        )
        colors[i] = rgb
    return colors


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple:
    """Convert HSV to RGB."""
    c = v * s
    x = c * (1.0 - abs((h * 6.0) % 2.0 - 1.0))
    m = v - c

    if h < 1.0 / 6.0:
        r, g, b = c, x, 0.0
    elif h < 2.0 / 6.0:
        r, g, b = x, c, 0.0
    elif h < 3.0 / 6.0:
        r, g, b = 0.0, c, x
    elif h < 4.0 / 6.0:
        r, g, b = 0.0, x, c
    elif h < 5.0 / 6.0:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x

    return (r + m, g + m, b + m)


def apply_mask_colors(
    arr: np.ndarray,
    mask_dict: dict,
    num_color_instances: int,
    background_value: float = 0.5,
) -> np.ndarray:
    """Apply colors to masked points. Both pred and GT use this for consistency.
    
    Args:
        arr: Point cloud array
        mask_dict: Dict mapping instance_id -> binary mask array
        num_color_instances: Number of colors to generate
        background_value: Gray value for non-masked points (0-1)
    
    Returns:
        (num_points, 3) color array with gray background and colored instances
    """
    mask_colors = np.ones((arr.shape[0], 3), dtype=np.float64) * background_value
    instance_colors = generate_instance_colors(num_color_instances)
    for instance_id, mask in mask_dict.items():
        if instance_id in instance_colors and len(mask) == arr.shape[0]:
            mask_colors[mask] = instance_colors[instance_id]
    return mask_colors


def class_color(class_id: int):
    # 1: rotation (muted orange), 2: translation (muted cyan)
    if class_id == 1:
        return np.array([0.93, 0.68, 0.42], dtype=np.float64)
    if class_id == 2:
        return np.array([0.42, 0.78, 0.90], dtype=np.float64)
    return np.array([0.76, 0.76, 0.76], dtype=np.float64)


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


def make_rotation_loop_arrow(
    origin: np.ndarray,
    axis: np.ndarray,
    color: np.ndarray,
    radius: float,
):
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return []
    axis = axis / norm

    tube_radius = max(radius * 0.12, 0.002)
    ring = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=radius,
        tube_radius=tube_radius,
        radial_resolution=28,
        tubular_resolution=20,
    )
    ring.compute_vertex_normals()
    ring.paint_uniform_color(color.tolist())

    # Place ring in the plane normal to articulation axis.
    rotation = _rotation_from_z(axis)
    ring.rotate(rotation, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    ring.translate(origin.astype(np.float64))

    # Arrowhead tangent to the ring to indicate rotational direction.
    phi = np.deg2rad(45.0)
    p_local = np.array([radius * np.cos(phi), radius * np.sin(phi), 0.0], dtype=np.float64)
    t_local = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=np.float64)
    p_world = origin + (rotation @ p_local)
    t_world = rotation @ t_local

    cone_h = max(radius * 0.32, 0.01)
    cone_r = max(radius * 0.16, 0.005)
    head = o3d.geometry.TriangleMesh.create_cone(radius=cone_r, height=cone_h)
    head.compute_vertex_normals()
    head.paint_uniform_color(color.tolist())
    head.rotate(_rotation_from_z(t_world / np.linalg.norm(t_world)), center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    head.translate(p_world.astype(np.float64))

    return [ring, head]


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
    draw_pred_masks: bool,
    draw_gt_masks: bool,
):
    mov_inter = "inter"
    input_data_dir = root_dir / f"data/processed/articulate3d_challenge_{mov_inter}" / split
    predictions_dir = root_dir / f"results/inference_{mov_inter}_seg/{mov_inter}_test"
    predicted_file = predictions_dir / f"{scene}.txt"
    gt_artic_file = input_data_dir / f"{scene}_articulation.h5"
    
    # Load database YAML to get correct paths
    db_yaml_path = root_dir / f"data/processed/articulate3d_challenge_{mov_inter}" / f"{split}_database.yaml"
    scene_entry = load_database_yaml(db_yaml_path, scene)
    gt_mask_file = None
    if scene_entry and "instance_gt_filepath" in scene_entry:
        gt_mask_file = root_dir / scene_entry["instance_gt_filepath"]

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

    # Axis length relative to scene size.
    xyz = arr[:, :3]
    scene_diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
    axis_length = max(scene_diag * 0.06, 0.04)
    marker_radius = max(scene_diag * 0.004, 0.01)

    # Initialize geometries based on which overlays we want
    if draw_pred_masks or draw_gt_masks:
        # If drawing masks, we'll color the point cloud based on masks
        # Start with a plain white point cloud that we'll color
        geometries = []
    else:
        # Otherwise start with the original scene point cloud
        pcd = build_point_cloud(arr)
        geometries = [pcd]
    pred_origins_for_stats = []
    gt_origins_for_stats = []

    if draw_pred or draw_pred_masks:
        pred_rows = parse_prediction_rows(predicted_file)
        pred_rows = [r for r in pred_rows if r["score"] >= score_thr]
        pred_rows.sort(key=lambda r: r["score"], reverse=True)
        if topk > 0:
            pred_rows = pred_rows[:topk]
    else:
        pred_rows = []

    if draw_pred_masks and pred_rows:
        print("Loading predicted masks...")
        pred_masks = load_predicted_masks_for_scene(predictions_dir, pred_rows)
        if pred_masks:
            print(f"Successfully loaded {len(pred_masks)} predicted masks")
            num_instances = max(max(pred_masks.keys()) + 1, 1) if pred_masks else 1
            mask_colors = apply_mask_colors(arr, pred_masks, num_instances)
            mask_pcd = o3d.geometry.PointCloud()
            mask_pcd.points = o3d.utility.Vector3dVector(
                arr[:, :3].astype(np.float64)
            )
            mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)
            geometries.append(mask_pcd)
            print(f"Visualizing predicted masks for {len(pred_masks)} instances")
        else:
            print("No predicted masks loaded")

    if draw_gt_masks:
        print("Loading GT masks...")
        if gt_mask_file is None:
            print("  GT mask file path not found in database")
        gt_labels = load_gt_masks(gt_mask_file)
        if gt_labels is not None:
            unique_labels = np.unique(gt_labels)
            # Create a binary mask dict using contiguous indices (0, 1, 2, ...)
            gt_mask_dict = {}
            color_idx = 0
            for label_id in unique_labels:
                if label_id != 0 and label_id != 255:  # Not background/ignore
                    gt_mask_dict[color_idx] = (gt_labels == label_id)
                    color_idx += 1
            num_instances = len(gt_mask_dict)
            print(f"  Found {num_instances} non-background GT instances")
            mask_colors = apply_mask_colors(arr, gt_mask_dict, max(num_instances, 1))
            gt_mask_pcd = o3d.geometry.PointCloud()
            gt_mask_pcd.points = o3d.utility.Vector3dVector(
                arr[:, :3].astype(np.float64)
            )
            gt_mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)
            geometries.append(gt_mask_pcd)
            print(f"  Visualizing GT masks")
        else:
            print("  GT mask file not found or could not be loaded")

    if draw_pred:
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
                if row["cls"] == 1:
                    geometries.extend(
                        make_rotation_loop_arrow(
                            origin,
                            axis,
                            color,
                            radius=max(axis_length * 0.35, marker_radius * 2.5),
                        )
                    )
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
                if row["cls"] == 1:
                    geometries.extend(
                        make_rotation_loop_arrow(
                            origin,
                            axis,
                            color,
                            radius=max(axis_length * 0.32, marker_radius * 2.6),
                        )
                    )
                gt_origins_for_stats.append(origin)
                gt_used += 1
        print(f"Loaded GT articulations: {len(gt_rows)}, visualized articulations: {gt_used}")

    if draw_pred:
        print("Pred color legend: rotation=orange (1, with circular arrow), translation=cyan (2)")
    if draw_gt:
        print("GT color legend: rotation=red (1, with circular arrow), translation=blue (2)")

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
    parser.add_argument(
        "--draw-pred-masks",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Draw predicted masks overlaid on point cloud (default: false)",
    )
    parser.add_argument(
        "--draw-gt-masks",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Draw GT masks overlaid on point cloud (default: false)",
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
        args.draw_pred_masks,
        args.draw_gt_masks,
    )