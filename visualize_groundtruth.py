"""
Visualize articulate3d ground truth annotations in 3D.

Loads raw scene data (parts.json, artic.json, mesh.ply) and displays:
- Segmented point cloud colored by part/object ID
- Inferred connectivity between parts (derived from label hierarchy)
- Articulation joints (origin markers + axis arrows + rotation loops)

Two dataset layouts are supported:

  Original articulate3d (full scenes):
    python visualize_groundtruth.py --scene-id 0a5c013435 \\
        --articulate3d-dir data/raw/articulate3d \\
        --scannet-dir <SCANNET_ROOT> \\
        --show-connectivity --show-joints

  Cropped OccArticulate3d (per-view subsets):
    python visualize_groundtruth.py --view-id 0a5c013435_1 \\
        --occ-dir data/raw/OccArticulate3d \\
        --show-connectivity --show-joints
"""

import os

import open3d as o3d
import numpy as np
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# ============================================================================
# Configuration: Set your ScanNet directory here
# ============================================================================
# Path to ScanNet root directory (parent of 'data' folder)
# get SCANNET_ROOT from environment variable if set
if "SCANNET_ROOT" in os.environ:
    SCANNET_ROOT = Path(os.environ["SCANNET_ROOT"])

# ============================================================================
# Color Utilities (adapted from visualize_results.py)
# ============================================================================

def generate_instance_colors(num_instances: int) -> Dict[int, np.ndarray]:
    """Generate distinct colors for instances using HSV space with golden ratio spacing."""
    colors = {}
    if num_instances == 0:
        return colors
    for i in range(num_instances):
        hue = (i * 0.618) % 1.0
        sat = 0.7 + (i % 3) * 0.1
        val = 0.75 + (i % 2) * 0.15
        rgb = np.array(_hsv_to_rgb(hue, sat, val), dtype=np.float64)
        colors[i] = rgb
    return colors


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
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


def articulation_color(artic_type: str) -> np.ndarray:
    """Color scheme for articulation types: rotation (red), translation (blue)."""
    if artic_type == "rotation":
        return np.array([1.0, 0.2, 0.2], dtype=np.float64)
    elif artic_type == "translation":
        return np.array([0.2, 0.45, 1.0], dtype=np.float64)
    else:
        return np.array([0.6, 0.6, 0.6], dtype=np.float64)


# ============================================================================
# Geometry Primitives (adapted from visualize_results.py)
# ============================================================================

def _rotation_from_z(axis_unit: np.ndarray) -> np.ndarray:
    """Compute rotation matrix that maps +Z to axis_unit."""
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v = np.cross(z, axis_unit)
    c = float(np.dot(z, axis_unit))
    s = float(np.linalg.norm(v))
    
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=np.float64,
        )
    
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


def make_axis_arrow(
    origin: np.ndarray, 
    axis: np.ndarray, 
    color: np.ndarray, 
    length: float
) -> Optional[o3d.geometry.TriangleMesh]:
    """Create a directional arrow along the articulation axis."""
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
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
    
    rotation = _rotation_from_z(axis)
    arrow.rotate(rotation, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    arrow.translate(origin.astype(np.float64))
    
    return arrow


def make_rotation_loop_arrow(
    origin: np.ndarray,
    axis: np.ndarray,
    color: np.ndarray,
    radius: float,
) -> List[o3d.geometry.TriangleMesh]:
    """Create a torus ring + arrowhead to indicate rotation."""
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
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
    
    rotation = _rotation_from_z(axis)
    ring.rotate(rotation, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    ring.translate(origin.astype(np.float64))
    
    # Arrowhead tangent to the ring
    # phi = np.deg2rad(45.0)
    # p_local = np.array([radius * np.cos(phi), radius * np.sin(phi), 0.0], dtype=np.float64)
    # t_local = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=np.float64)
    # p_world = origin + (rotation @ p_local)
    # t_world = rotation @ t_local
    
    # cone_h = max(radius * 0.32, 0.01)
    # cone_r = max(radius * 0.16, 0.005)
    # head = o3d.geometry.TriangleMesh.create_cone(radius=cone_r, height=cone_h)
    # head.compute_vertex_normals()
    # head.paint_uniform_color(color.tolist())
    # head.rotate(
    #     _rotation_from_z(t_world / np.linalg.norm(t_world)),
    #     center=np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # )
    # head.translate(p_world.astype(np.float64))
    
    return [ring] # , head


def make_origin_marker(origin: np.ndarray, color: np.ndarray, radius: float) -> o3d.geometry.TriangleMesh:
    """Create a small sphere at the joint origin."""
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sph.compute_vertex_normals()
    sph.paint_uniform_color(color.tolist())
    sph.translate(origin.astype(np.float64))
    return sph


# ============================================================================
# Data Loaders
# ============================================================================

def load_parts_annotation(parts_file: Path) -> Dict:
    """Load parts.json and return parsed annotation dict."""
    with open(parts_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_articulation_annotation(artic_file: Path) -> Dict:
    """Load artic.json and return parsed annotation dict."""
    with open(artic_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_mesh(mesh_file: Path) -> o3d.geometry.TriangleMesh:
    """Load mesh from PLY file."""
    mesh = o3d.io.read_triangle_mesh(str(mesh_file))
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color([0.5, 0.5, 0.5])
    return mesh


# ============================================================================
# Annotation Processing
# ============================================================================

def process_parts_annotations(parts_data: Dict) -> Tuple[Dict, Dict, Dict]:
    """
    Parse parts annotations and build lookup maps.
    
    Returns:
        part_id_to_anno: partId -> annotation entry
        object_id_to_parts: objectId -> list of partIds
        part_id_to_verts: partId -> list of vertex indices
    """
    part_id_to_anno = {}
    object_id_to_parts = defaultdict(list)
    part_id_to_verts = defaultdict(list)
    
    if 'data' not in parts_data or 'annotations' not in parts_data['data']:
        print("Warning: no annotations found in parts.json")
        return part_id_to_anno, object_id_to_parts, part_id_to_verts
    
    for anno in parts_data['data']['annotations']:
        part_id = anno.get('partId')
        object_id = anno.get('objectId')
        vert_indices = anno.get('vertIndices', [])
        
        if part_id is not None:
            part_id_to_anno[part_id] = anno
            if object_id is not None:
                object_id_to_parts[object_id].append(part_id)
            part_id_to_verts[part_id] = vert_indices
    
    return part_id_to_anno, object_id_to_parts, part_id_to_verts


def process_articulation_annotations(artic_data: Dict) -> Dict:
    """
    Parse articulation annotations.
    
    Returns:
        part_id_to_artic: partId -> articulation entry with normalized axis
    """
    part_id_to_artic = {}
    
    if 'data' not in artic_data or 'articulations' not in artic_data['data']:
        print("Warning: no articulations found in artic.json")
        return part_id_to_artic
    
    for artic in artic_data['data']['articulations']:
        pid = artic.get('pid')
        
        if pid is not None:
            # Normalize axis
            axis = np.array(artic.get('axis', [0, 0, 1]), dtype=np.float64)
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 1e-9:
                axis = axis / axis_norm
            else:
                print(f"Warning: zero-length axis for pid {pid}, using default [0, 0, 1]")
                axis = np.array([0.0, 0.0, 1.0])
            
            artic_entry = artic.copy()
            artic_entry['axis'] = axis
            part_id_to_artic[pid] = artic_entry
    
    return part_id_to_artic


# ============================================================================
# Segmentation & Coloring
# ============================================================================

def build_segmented_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    part_id_to_verts: Dict[int, List[int]],
    part_id_to_anno: Dict[int, Dict],
    color_mode: str = "part"
) -> o3d.geometry.PointCloud:
    """
    Convert mesh to point cloud with per-vertex colors based on part assignment.
    
    Args:
        mesh: Input triangle mesh
        part_id_to_verts: Mapping from part ID to list of vertex indices
        part_id_to_anno: Mapping from part ID to annotation dict (for object_id)
        color_mode: "part" (by partId) or "object" (by objectId)
    
    Returns:
        Colored point cloud
    """
    # Convert mesh to point cloud
    pcd = o3d.geometry.PointCloud()
    vertices = np.array(mesh.vertices)
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # Initialize colors (background gray for unannotated vertices)
    colors = np.full((len(vertices), 3), 0.5, dtype=np.float64)
    
    # Build vertex-to-part mapping
    vertex_to_part = {}
    for part_id, vert_indices in part_id_to_verts.items():
        for vid in vert_indices:
            if 0 <= vid < len(vertices):
                vertex_to_part[vid] = part_id
    
    # Determine color mapping
    if color_mode == "object":
        # Group by object ID
        object_id_to_parts = defaultdict(list)
        for part_id, anno in part_id_to_anno.items():
            object_id = anno.get('objectId')
            if object_id is not None:
                object_id_to_parts[object_id].append(part_id)
        
        unique_object_ids = sorted([oid for oid in object_id_to_parts.keys() if oid is not None])
        object_colors = generate_instance_colors(len(unique_object_ids))
        
        for vid, part_id in vertex_to_part.items():
            if part_id in part_id_to_anno:
                object_id = part_id_to_anno[part_id].get('objectId')
                if object_id is not None and object_id in unique_object_ids:
                    color_idx = unique_object_ids.index(object_id)
                    colors[vid] = object_colors[color_idx]
    elif color_mode == "original":
        colors = mesh.vertex_colors 
    else:
        # Color by part ID (default)
        unique_part_ids = sorted([pid for pid in part_id_to_anno.keys()])
        part_colors = generate_instance_colors(len(unique_part_ids))
        
        for vid, part_id in vertex_to_part.items():
            if part_id in unique_part_ids:
                color_idx = unique_part_ids.index(part_id)
                colors[vid] = part_colors[color_idx]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ============================================================================
# Connectivity Inference
# ============================================================================

def infer_label_hierarchy(part_id_to_anno: Dict[int, Dict]) -> Dict[int, Optional[int]]:
    """
    Infer parent-child relationships from hierarchical label structure.
    
    Labels like "door.1.1", "door.1.1.2_1" encode hierarchy via dot/underscore separators.
    This function extracts parent-child edges.
    
    Returns:
        parent_map: part_id -> parent_part_id (or None if root)
    """
    def extract_hierarchy_code(label: str) -> str:
        """Extract numeric hierarchy code from label (e.g., '.1.1.2_1')."""
        match = re.search(r'[\d._]+$', label)
        return match.group() if match else ''

    def get_parent_hierarchy(hierarchy_code: str) -> Optional[str]:
        """Get parent code by removing the last numeric component."""
        if not hierarchy_code:
            return None

        parts = hierarchy_code.split('.')
        if len(parts) <= 2:
            return None

        return '.'.join(parts[:-1])

    hierarchy_to_part_id = {}
    for part_id, anno in part_id_to_anno.items():
        label = anno.get('label', '')
        hierarchy_code = extract_hierarchy_code(label)
        if hierarchy_code:
            hierarchy_to_part_id[hierarchy_code] = part_id

    parent_map = {}
    for part_id, anno in part_id_to_anno.items():
        label = anno.get('label', '')
        hierarchy_code = extract_hierarchy_code(label)
        parent_part_id = None

        if hierarchy_code:
            parent_hierarchy = get_parent_hierarchy(hierarchy_code)
            if parent_hierarchy and parent_hierarchy in hierarchy_to_part_id:
                parent_part_id = hierarchy_to_part_id[parent_hierarchy]

        parent_map[part_id] = parent_part_id

    return parent_map


def build_connectivity_edges(
    part_id_to_anno: Dict[int, Dict],
    parent_map: Dict[int, Optional[int]],
    part_id_to_verts: Dict[int, List[int]],
    vertices: np.ndarray
) -> List[Tuple[int, int]]:
    """
    Build connectivity edges from parent-child relationships.
    
    Returns:
        List of (parent_part_id, child_part_id) tuples
    """
    edges = []
    
    for part_id, parent_id in parent_map.items():
        if parent_id is not None:
            edges.append((parent_id, part_id))
    
    return edges


def get_part_centroid(
    part_id: int,
    part_id_to_anno: Dict[int, Dict],
    part_id_to_verts: Dict[int, List[int]],
    vertices: np.ndarray
) -> np.ndarray:
    """Compute centroid of a part using either OBB centroid or vertex centroid."""
    if part_id in part_id_to_anno:
        anno = part_id_to_anno[part_id]
        obb = anno.get('obb')
        if isinstance(obb, dict) and 'centroid' in obb:
            centroid = np.asarray(obb['centroid'], dtype=np.float64)
            if centroid.shape == (3,):
                return centroid

    vert_indices = part_id_to_verts.get(part_id, [])
    valid_indices = [vid for vid in vert_indices if 0 <= vid < len(vertices)]
    if valid_indices:
        return vertices[valid_indices].mean(axis=0)

    return np.zeros(3, dtype=np.float64)


def build_part_color_lookup(
    part_id_to_anno: Dict[int, Dict],
    color_mode: str
) -> Dict[int, np.ndarray]:
    """Build deterministic part color lookup matching the point-cloud coloring strategy."""
    part_colors: Dict[int, np.ndarray] = {}

    if color_mode == "object":
        object_id_to_parts = defaultdict(list)
        for part_id, anno in part_id_to_anno.items():
            object_id = anno.get("objectId")
            if object_id is not None:
                object_id_to_parts[object_id].append(part_id)

        unique_object_ids = sorted([oid for oid in object_id_to_parts.keys() if oid is not None])
        object_colors = generate_instance_colors(len(unique_object_ids))
        object_to_color = {
            oid: object_colors[i]
            for i, oid in enumerate(unique_object_ids)
        }

        for part_id, anno in part_id_to_anno.items():
            object_id = anno.get("objectId")
            if object_id in object_to_color:
                part_colors[part_id] = object_to_color[object_id]
            else:
                part_colors[part_id] = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    else:
        unique_part_ids = sorted(part_id_to_anno.keys())
        generated = generate_instance_colors(len(unique_part_ids))
        for i, part_id in enumerate(unique_part_ids):
            part_colors[part_id] = generated[i]

    return part_colors


def make_line_tube(
    start: np.ndarray,
    end: np.ndarray,
    color: np.ndarray,
    radius: float
) -> Optional[o3d.geometry.TriangleMesh]:
    """Create a thick cylinder segment between two 3D points."""
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9:
        return None

    axis = direction / length
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(color.tolist())

    rotation = _rotation_from_z(axis)
    cylinder.rotate(rotation, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
    midpoint = (start + end) * 0.5
    cylinder.translate(midpoint.astype(np.float64))
    return cylinder


def draw_connectivity_edges(
    edges: List[Tuple[int, int]],
    part_id_to_anno: Dict[int, Dict],
    part_id_to_verts: Dict[int, List[int]],
    vertices: np.ndarray,
    color_mode: str,
    scene_diag: float
) -> List[o3d.geometry.Geometry3D]:
    """Create thick, gradient-colored 3D tubes with directional arrows for parent-child connectivity edges."""
    if not edges:
        return []

    part_color_lookup = build_part_color_lookup(part_id_to_anno, color_mode)
    geometries: List[o3d.geometry.Geometry3D] = []
    tube_radius = 0.01
    gradient_steps = 10
    arrow_cone_radius = tube_radius * 3.0
    arrow_cone_height = tube_radius * 6.0

    for parent_id, child_id in edges:
        if parent_id not in part_id_to_anno or child_id not in part_id_to_anno:
            continue

        parent_centroid = get_part_centroid(parent_id, part_id_to_anno, part_id_to_verts, vertices)
        child_centroid = get_part_centroid(child_id, part_id_to_anno, part_id_to_verts, vertices)

        parent_color = part_color_lookup.get(parent_id, np.array([0.6, 0.6, 0.6], dtype=np.float64))
        child_color = part_color_lookup.get(child_id, np.array([0.6, 0.6, 0.6], dtype=np.float64))

        # Draw gradient-colored tube segments from parent to child
        for i in range(gradient_steps):
            t0 = i / gradient_steps
            t1 = (i + 1) / gradient_steps
            seg_start = (1.0 - t0) * parent_centroid + t0 * child_centroid
            seg_end = (1.0 - t1) * parent_centroid + t1 * child_centroid
            tm = 0.5 * (t0 + t1)
            seg_color = (1.0 - tm) * parent_color + tm * child_color

            segment = make_line_tube(seg_start, seg_end, seg_color, tube_radius)
            if segment is not None:
                geometries.append(segment)

        # Add arrow cone at child endpoint to show direction
        direction = child_centroid - parent_centroid
        dir_length = float(np.linalg.norm(direction))
        if dir_length > 1e-9:
            dir_normalized = direction / dir_length
            arrow_origin = child_centroid - dir_normalized * (arrow_cone_height * 0.5)
            
            cone = o3d.geometry.TriangleMesh.create_cone(
                radius=arrow_cone_radius,
                height=arrow_cone_height
            )
            cone.compute_vertex_normals()
            cone.paint_uniform_color(child_color.tolist())
            
            cone_rotation = _rotation_from_z(dir_normalized)
            cone.rotate(cone_rotation, center=np.array([0.0, 0.0, 0.0], dtype=np.float64))
            cone.translate(arrow_origin.astype(np.float64))
            geometries.append(cone)

    return geometries

def render_articulations(
    part_id_to_artic: Dict[int, Dict],
    part_id_to_anno: Dict[int, Dict],
    scene_diag: float
) -> List[o3d.geometry.Geometry3D]:
    """Render all articulation joints in the scene."""
    geometries = []
    
    axis_length = max(scene_diag * 0.06, 0.04)
    marker_radius = max(scene_diag * 0.004, 0.01)
    loop_radius = max(axis_length * 0.35, marker_radius * 2.5)
    
    for pid, artic in part_id_to_artic.items():
        if pid not in part_id_to_anno:
            print(f"Warning: articulation pid {pid} not found in parts annotations")
            continue
        
        origin = np.array(artic.get('origin', [0, 0, 0]), dtype=np.float64)
        axis = artic['axis']  # Already normalized
        artic_type = artic.get('type', 'rotation')
        color = articulation_color(artic_type)
        
        # Draw origin marker
        marker = make_origin_marker(origin, color, marker_radius)
        geometries.append(marker)
        
        # Draw axis arrow
        arrow = make_axis_arrow(origin, axis, color, axis_length)
        if arrow is not None:
            geometries.append(arrow)
        
        # Draw rotation loop if applicable
        if artic_type == 'rotation':
            loop_geoms = make_rotation_loop_arrow(origin, axis, color, loop_radius)
            geometries.extend(loop_geoms)
    
    return geometries


# ============================================================================
# Main Visualization
# ============================================================================

def _visualize_from_files(
    parts_file: Path,
    artic_file: Path,
    mesh_file: Path,
    label: str,
    color_mode: str = "part",
    show_connectivity: bool = True,
    show_joints: bool = True,
    verbose: bool = True,
):
    """Core visualization — accepts resolved file paths directly."""
    # Validate existence
    errors = []
    if not parts_file.exists():
        errors.append(f"Parts file not found: {parts_file}")
    if not artic_file.exists():
        errors.append(f"Articulation file not found: {artic_file}")
    if not mesh_file.exists():
        errors.append(f"Mesh file not found: {mesh_file}")

    if errors:
        print("Error: Missing required files:")
        for err in errors:
            print(f"  {err}")
        return

    if verbose:
        print(f"Loading {label}")
        print(f"  Parts: {parts_file}")
        print(f"  Articulations: {artic_file}")
        print(f"  Mesh: {mesh_file}")

    # Load annotations
    parts_data = load_parts_annotation(parts_file)
    artic_data = load_articulation_annotation(artic_file)
    mesh = load_mesh(mesh_file)

    # Process annotations
    part_id_to_anno, object_id_to_parts, part_id_to_verts = process_parts_annotations(parts_data)
    part_id_to_artic = process_articulation_annotations(artic_data)

    if verbose:
        print(f"Loaded {len(part_id_to_anno)} parts and {len(part_id_to_artic)} articulations")

    # Build segmented point cloud
    pcd = build_segmented_point_cloud(mesh, part_id_to_verts, part_id_to_anno, color_mode)
    geometries = [pcd]

    # Compute scene diagonal for scaling
    vertices = np.array(mesh.vertices)
    scene_diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))

    # Draw connectivity
    if show_connectivity:
        parent_map = infer_label_hierarchy(part_id_to_anno)
        edges = build_connectivity_edges(part_id_to_anno, parent_map, part_id_to_verts, vertices)
        conn_geometries = draw_connectivity_edges(
            edges,
            part_id_to_anno,
            part_id_to_verts,
            vertices,
            color_mode,
            scene_diag,
        )
        if conn_geometries:
            geometries.extend(conn_geometries)
            if verbose:
                print(f"Inferred {len(edges)} connectivity edges from label hierarchy")

    # Draw articulations
    if show_joints:
        joint_geoms = render_articulations(part_id_to_artic, part_id_to_anno, scene_diag)
        geometries.extend(joint_geoms)
        if verbose:
            print(f"Rendered {len(part_id_to_artic)} articulation joints")

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print(f"Scene: {label}")
        print(f"Total vertices: {len(vertices)}")
        print(f"Annotated parts: {len(part_id_to_anno)}")
        print(f"Objects: {len(object_id_to_parts)}")
        print(f"Articulations: {len(part_id_to_artic)}")
        print(f"Scene bounds: {vertices.min(axis=0)} to {vertices.max(axis=0)}")
        print("="*60)
        print("\nColor legend:")
        if color_mode == "part":
            print("  - Each distinct color represents a different part")
        else:
            print("  - Each distinct color represents a different object")
        print("  - Gray: unannotated vertices")
        if show_joints:
            print("\nArticulation colors:")
            print("  - Red: rotation joints")
            print("  - Blue: translation joints")
            print("  - Circular ring: indicates rotational motion")

    # Launch viewer
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Articulate3D Ground Truth: {label}",
        width=1280,
        height=960,
    )


def visualize_articulate3d_scene(
    scene_id: str,
    articulate3d_dir: Path,
    scannet_dir: Path,
    color_mode: str = "part",
    show_connectivity: bool = True,
    show_joints: bool = True,
    verbose: bool = True,
):
    """Visualize a full articulate3d scene (original dataset layout)."""
    parts_file = articulate3d_dir / f"{scene_id}_parts.json"
    artic_file = articulate3d_dir / f"{scene_id}_artic.json"
    mesh_file = scannet_dir / "data" / scene_id / "scans" / "mesh_aligned_0.05.ply"
    _visualize_from_files(
        parts_file, artic_file, mesh_file, scene_id,
        color_mode=color_mode,
        show_connectivity=show_connectivity,
        show_joints=show_joints,
        verbose=verbose,
    )


def visualize_occ_scene(
    view_id: str,
    occ_dir: Path,
    color_mode: str = "part",
    show_connectivity: bool = True,
    show_joints: bool = True,
    verbose: bool = True,
):
    """Visualize a cropped OccArticulate3d view.

    Expects the structured layout produced by crop_groundtruth.py:
        occ_dir/scans/{view_id}/{view_id}_parts.json
        occ_dir/scans/{view_id}/{view_id}_artic.json
        occ_dir/scans/{view_id}/mesh_aligned_0.05.ply
    """
    scan_dir = occ_dir / "scans" / view_id
    parts_file = scan_dir / f"{view_id}_parts.json"
    artic_file = scan_dir / f"{view_id}_artic.json"
    mesh_file = scan_dir / "mesh_aligned_0.05.ply"
    _visualize_from_files(
        parts_file, artic_file, mesh_file, view_id,
        color_mode=color_mode,
        show_connectivity=show_connectivity,
        show_joints=show_joints,
        verbose=verbose,
    )


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize articulate3d ground truth annotations in 3D"
    )

    # --- dataset selection (mutually exclusive) ---
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--scene-id",
        type=str,
        metavar="SCENE_ID",
        help="Full-scene ID for original articulate3d layout (e.g. 0a5c013435)",
    )
    mode_group.add_argument(
        "--view-id",
        type=str,
        metavar="VIEW_ID",
        help="View ID for cropped OccArticulate3d layout (e.g. 0a5c013435_1)",
    )

    # --- original articulate3d paths ---
    parser.add_argument(
        "--articulate3d-dir",
        type=Path,
        default=Path("data/raw/articulate3d"),
        help="Path to articulate3d raw data directory (used with --scene-id)",
    )
    parser.add_argument(
        "--scannet-dir",
        type=Path,
        required=False,
        help="Path to ScanNet root (parent of 'data' folder, used with --scene-id)",
    )

    # --- OccArticulate3d paths ---
    parser.add_argument(
        "--occ-dir",
        type=Path,
        default=Path("data/raw/OccArticulate3d"),
        help="Path to OccArticulate3d dataset root (used with --view-id)",
    )

    # --- display options ---
    parser.add_argument(
        "--color-mode",
        type=str,
        choices=["part", "object", "original"],
        default="original",
        help="Segmentation coloring mode (default: part)",
    )
    parser.add_argument(
        "--show-connectivity",
        action="store_true",
        default=True,
        help="Draw inferred part connectivity edges",
    )
    parser.add_argument(
        "--hide-connectivity",
        action="store_false",
        dest="show_connectivity",
        help="Hide connectivity edges",
    )
    parser.add_argument(
        "--show-joints",
        action="store_true",
        default=True,
        help="Draw articulation joints",
    )
    parser.add_argument(
        "--hide-joints",
        action="store_false",
        dest="show_joints",
        help="Hide articulation overlays",
    )

    args = parser.parse_args()

    kwargs = dict(
        color_mode=args.color_mode,
        show_connectivity=args.show_connectivity,
        show_joints=args.show_joints,
        verbose=True,
    )

    if args.view_id:
        # --- OccArticulate3d (cropped) ---
        visualize_occ_scene(
            view_id=args.view_id,
            occ_dir=args.occ_dir.resolve(),
            **kwargs,
        )
    else:
        # --- original articulate3d ---
        scannet_dir = args.scannet_dir
        if scannet_dir is None:
            candidates = [
                SCANNET_ROOT,
                Path("ScanNet"),
                Path("..") / "ScanNet",
                Path("/mnt/Data/ScanNet"),
                Path.home() / "data" / "ScanNet",
            ]
            for cand in candidates:
                if (cand / "data").exists():
                    scannet_dir = cand
                    print(f"Using ScanNet directory: {scannet_dir}")
                    break

        if scannet_dir is None:
            print("Error: ScanNet directory not found.")
            print("  - Set SCANNET_ROOT environment variable")
            print("  - Or specify --scannet-dir on command line")
            return

        visualize_articulate3d_scene(
            scene_id=args.scene_id,
            articulate3d_dir=args.articulate3d_dir.resolve(),
            scannet_dir=Path(scannet_dir).resolve(),
            **kwargs,
        )


if __name__ == "__main__":
    main()
