import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from jaxtyping import Bool, Float, Int

if "SCANNET_ROOT" in os.environ:
    SCANNET_ROOT = Path(os.environ["SCANNET_ROOT"])
else:
    raise RuntimeError("Download scannet and set SCANNET_ROOT environment variable")

if "ARTICULATE_3D_DATA_ROOT" in os.environ:
    ARTICULATE_3D_DATA_ROOT = Path(os.environ["ARTICULATE_3D_DATA_ROOT"])
else:
    raise RuntimeError(
        "Download articulate_3d and set ARTICULATE_3D_DATA_ROOT environment variable"
    )


@dataclass
class OccludedViewResult:
    """Result of generating an occluded view."""

    scene_id: str
    visible_mesh_indices: Int[np.ndarray, "M"]  # (M,) indices into mesh.vertices
    visible_triangle_indices: (
        np.ndarray
    )  # (Ntri,) indices into mesh.triangles where all 3 vertices are visible
    orientation_quat: np.ndarray  # (4,) quaternion
    intrinsics: np.ndarray  # (3, 3) camera matrix
    range: float
    camera_pos: Float[np.ndarray, "3"]  # (3,) world coordinates
    is_valid: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "camera_pos": self.camera_pos.tolist(),
            "visible_mesh_indices": self.visible_mesh_indices.tolist(),
            "visible_triangle_indices": self.visible_triangle_indices.tolist(),
            "orientation_quat": self.orientation_quat.tolist(),
            "intrinsics": self.intrinsics.tolist(),
            "range": float(self.range),
        }


class OccludedArticulate3DCreator:
    scene_id: str
    gt_parts_json: Path
    gt_artic_json: Path
    mesh_path: Path
    mesh: o3d.t.geometry.TriangleMesh
    raycast_scene: o3d.t.geometry.RaycastingScene

    def __init__(self, scene_id: str):
        self.scene_id = scene_id
        self.gt_parts_json, self.gt_artic_json, self.mesh_path = self._find_data()

        self.mesh = o3d.t.io.read_triangle_mesh(
            self.mesh_path.absolute().as_posix()
        ).cpu()
        self.mesh = self.mesh.compute_vertex_normals()

        self.raycast_scene = o3d.t.geometry.RaycastingScene()
        self.raycast_scene.add_triangles(self.mesh)

    def _find_data(self) -> tuple[Path, Path, Path]:
        gt_parts_json = ARTICULATE_3D_DATA_ROOT / (self.scene_id + "_parts.json")
        gt_artic_json = ARTICULATE_3D_DATA_ROOT / (self.scene_id + "_artic.json")
        mesh = SCANNET_ROOT / "data" / self.scene_id / "scans" / "mesh_aligned_0.05.ply"

        if not gt_parts_json.exists():
            raise FileNotFoundError(f"Could not find {gt_parts_json}")
        if not gt_artic_json.exists():
            raise FileNotFoundError(f"Could not find {gt_artic_json}")
        if not mesh.exists():
            raise FileNotFoundError(f"Could not find {mesh}")

        return gt_parts_json, gt_artic_json, mesh

    @staticmethod
    def _classify_interior_points(
        points: Float[np.ndarray, "N 3"],
        mesh: o3d.geometry.TriangleMesh,
        k_nearest: int = 25,
    ) -> Bool[np.ndarray, "N"]:
        """
        Simple heuristic inside/outside test using:
        - nearest vertex (Open3D KDTree)
        - vertex normals
        """

        vertices: Float[np.ndarray, "V 3"] = mesh.vertex.positions.numpy()
        normals: Float[np.ndarray, "V 3"] = mesh.vertex.normals.numpy()

        # Build Open3D point cloud (KDTree is internal to Open3D)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        kdtree = o3d.geometry.KDTreeFlann(pcd)

        inside = np.zeros((points.shape[0],), dtype=bool)

        for i in range(points.shape[0]):
            p: Float[np.ndarray, "3"] = points[i]

            # nearest vertex
            _, idx, _ = kdtree.search_knn_vector_3d(p, k_nearest)
            v: Float[np.ndarray, "3"] = vertices[idx]
            n: Float[np.ndarray, "3"] = normals[idx]

            # signed test
            inside[i] = np.all(np.matmul(p - v, n.T) >= 0)

        return inside

    def sample_inside_points(self, spacing: float = 0.2) -> Float[np.ndarray, "N 3"]:
        if self.mesh is None:
            raise ValueError("Mesh is not loaded")

        min_bound = self.mesh.get_min_bound().numpy()
        max_bound = self.mesh.get_max_bound().numpy()

        x = np.arange(min_bound[0], max_bound[0], spacing)
        y = np.arange(min_bound[1], max_bound[1], spacing)
        z = np.arange(min_bound[2], max_bound[2], spacing)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

        points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
        inside = OccludedArticulate3DCreator._classify_interior_points(
            points, self.mesh
        )

        return points[inside]

    def _select_from_view(
        self,
        view_pos: Float[np.ndarray, "3"],
        orientation: Float[np.ndarray, "3 3"],
        range: float,
        intrinsics: Float[np.ndarray, "3 3"],
    ) -> Float[np.ndarray, "N 3"]:
        """
        Select mesh vertices inside camera frustum using projection.
        """

        if self.mesh is None:
            raise ValueError("Mesh is not loaded")

        vertices: Float[np.ndarray, "V 3"] = self.mesh.vertex.positions.numpy()

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # ---- transform to camera coordinates ----
        R = orientation
        t = view_pos

        # world -> camera
        cam_pts: Float[np.ndarray, "V 3"] = (R.T @ (vertices - t).T).T

        x, y, z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]

        # keep points in front of camera and within range
        valid_depth: Bool[np.ndarray, "V"] = (z > 0) & (z < range)

        # project to pixel space
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        width = 2 * cx
        height = 2 * cy

        in_frustum: Bool[np.ndarray, "V"] = (
            valid_depth & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        )

        return vertices[in_frustum]

    def get_visible_points_from_view(
        self,
        camera_pos: Float[np.ndarray, "3"],
        orientation: Float[np.ndarray, "3 3"],
        range: float,
        intrinsics: Float[np.ndarray, "3 3"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get visible point indices from a given camera view using raycasting.

        Uses the mesh representation and raycasting to determine which points
        are occluded by the mesh geometry.

        Args:
            camera_pos: Camera position in world coordinates (3,)
            orientation: Camera orientation matrix (3, 3)
            range: Maximum depth range for visible points
            intrinsics: Camera intrinsic matrix (3, 3)

        Returns:
            Array of visible vertex indices (M,) into self.mesh.vertices
        """
        if self.mesh is None:
            raise ValueError("Mesh is not loaded")

        # First, select points in the camera frustum
        vertices: Float[np.ndarray, "V 3"] = self.mesh.vertex.positions.numpy()

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Transform to camera coordinates
        R = orientation
        t = camera_pos

        # world -> camera
        cam_pts: Float[np.ndarray, "V 3"] = (R.T @ (vertices - t).T).T

        x, y, z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]

        # Keep points in front of camera and within range
        valid_depth: Bool[np.ndarray, "V"] = (z > 0) & (z < range)

        # Project to pixel space
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        width = 2 * cx
        height = 2 * cy

        in_frustum: Bool[np.ndarray, "V"] = (
            valid_depth & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        )

        frustum_indices = np.where(in_frustum)[0]  # Indices into original mesh vertices
        frustum_vertices = vertices[in_frustum]

        if len(frustum_vertices) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        # Now perform visibility check using raycasting
        num_points = frustum_vertices.shape[0]

        # Compute ray directions: from camera to each point
        # Ensure camera_pos is properly shaped as (3,) for numpy broadcasting
        camera_pos = np.asarray(camera_pos, dtype=np.float64).reshape(3)
        ray_dirs = frustum_vertices - camera_pos  # (N, 3)
        ray_distances = np.linalg.norm(ray_dirs, axis=1)  # (N,)

        # Normalize ray directions
        ray_dirs_normalized = ray_dirs / (ray_distances[:, np.newaxis] + 1e-8)

        # Convert to tensor
        rays_tensor = o3d.core.Tensor(
            np.hstack(
                [
                    np.tile(
                        camera_pos, (num_points, 1)
                    ),  # ray origins (camera position for all)
                    ray_dirs_normalized,  # normalized ray directions
                ]
            ),
            dtype=o3d.core.Dtype.Float32,
        )

        # Cast rays and get closest intersections
        ans = self.raycast_scene.cast_rays(rays_tensor)
        hit_distances = ans["t_hit"].numpy()  # Distance to closest intersection

        # A point is visible if:
        # - No intersection was found (hit_distance is inf/very large), OR
        # - The intersection is at or beyond the point's distance (with small epsilon for numerical stability)
        epsilon = 1e-4
        visible_mask = hit_distances >= ray_distances - epsilon

        visible_indices = frustum_indices[visible_mask]

        visible_set = set(visible_indices.tolist())
        triangles = self.mesh.triangle.indices.numpy()  # (F, 3)
        visible_triangle_indices = np.array(
            [
                i
                for i, tri in enumerate(triangles)
                if tri[0] in visible_set
                and tri[1] in visible_set
                and tri[2] in visible_set
            ],
            dtype=np.int64,
        )

        return visible_indices, visible_triangle_indices

    def validate_view(self, result: OccludedViewResult) -> bool:
        """Validate whether an occluded view is acceptable.

        Currently a placeholder that always returns True.
        Can be extended to check:
        - Minimum number of visible points
        - Point cloud density
        - Viewing angle quality
        - etc.

        Args:
            result: OccludedViewResult to validate

        Returns:
            True if view is valid, False otherwise
        """
        return result.visible_mesh_indices.size > 1000

    def generate_view(
        self,
        camera_pos: Float[np.ndarray, "3"],
        orientation_quat: np.ndarray,
        intrinsics: dict[str, float],
        range: float,
    ) -> OccludedViewResult:
        """Generate a single occluded view.

        Args:
            camera_pos_idx: Index into self.observation_points
            orientation_quat: Quaternion (4,) for camera orientation
            intrinsics: Dict with keys fx, fy, cx, cy
            range: Maximum depth range

        Returns:
            OccludedViewResult with generated indices and metadata
        """
        # Ensure camera_pos is a proper 1D array with shape (3,)
        camera_pos = camera_pos.reshape(3)
        orientation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(
            orientation_quat
        )

        intrinsics_matrix = np.array(
            [
                [intrinsics["fx"], 0, intrinsics["cx"]],
                [0, intrinsics["fy"], intrinsics["cy"]],
                [0, 0, 1],
            ]
        )

        visible_indices, visible_triangle_indices = self.get_visible_points_from_view(
            camera_pos,
            orientation_matrix,
            range,
            intrinsics_matrix,
        )

        result = OccludedViewResult(
            scene_id=self.scene_id,
            visible_mesh_indices=visible_indices,
            visible_triangle_indices=visible_triangle_indices,
            orientation_quat=orientation_quat,
            intrinsics=intrinsics_matrix,
            range=range,
            camera_pos=camera_pos,
        )

        result.is_valid = self.validate_view(result)
        return result

    def visualize(
        self,
        result: OccludedViewResult,
    ) -> None:
        """Visualize mesh and visible points for a given view.

        Args:
            result: OccludedViewResult to visualize
        """
        # draw original mesh as pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.mesh.vertex.positions.numpy())
        pcd.colors = o3d.utility.Vector3dVector(self.mesh.vertex.colors.numpy())

        # draw visible points in red
        visible_pcd = o3d.geometry.PointCloud()
        visible_points = self.mesh.vertex.positions.numpy()[result.visible_mesh_indices]
        visible_pcd.points = o3d.utility.Vector3dVector(visible_points)
        visible_pcd.paint_uniform_color([0, 1, 0])

        # load mesh as triangle mesh for visualization
        mesh_vis = o3d.geometry.TriangleMesh()
        mesh_vis.vertices = pcd.points
        mesh_vis.triangles = o3d.utility.Vector3iVector(
            self.mesh.triangle.indices.numpy()[result.visible_triangle_indices, :]
        )
        mesh_vis.paint_uniform_color([1, 0, 0])

        # draw camera position and viewing arrow
        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        camera_sphere.translate(result.camera_pos)
        camera_sphere.paint_uniform_color([0, 1, 0])

        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.02,
            cone_radius=0.04,
            cylinder_height=0.2,
            cone_height=0.05,
        )
        arrow = arrow.rotate(
            o3d.geometry.get_rotation_matrix_from_quaternion(result.orientation_quat),
            center=(np.zeros(3)),
        )
        arrow = arrow.translate(result.camera_pos)
        arrow.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries(
            [pcd, visible_pcd, mesh_vis, camera_sphere, arrow]
        )  # type: ignore
