"""Ground-truth depth and surface-normal rendering by exact ray casting.

Camera convention (OpenCV): +x right, +y down, +z forward. Pixel (u, v) has its
center at (u + 0.5, v + 0.5). For every pixel we cast the ray through its
center and report:

  depth   (H, W)     z-depth in the camera frame (0 where the ray misses)
  normals (H, W, 3)  unit surface normal (N_x, N_y, N_z) in the camera frame
  mask    (H, W)     True where the ray hits the surface

Mesh normals are barycentrically interpolated vertex normals (Phong normals),
so smooth surfaces get a continuous normal map instead of per-facet shading.
"""

import math
from dataclasses import dataclass
from typing import Callable

import torch

from geometry import Mesh

# (origin (3,), directions (R, 3)) -> (t (R,), world normals (R, 3)); t = inf on miss.
Intersector = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Camera:
    K: torch.Tensor  # (3, 3) intrinsics
    world_to_cam: torch.Tensor  # (4, 4) extrinsics [R | t]
    width: int
    height: int

    @property
    def center(self) -> torch.Tensor:
        R, t = self.world_to_cam[:3, :3], self.world_to_cam[:3, 3]
        return -R.T @ t


def intrinsics(width: int, height: int, fov_y_deg: float) -> torch.Tensor:
    """Pinhole K with square pixels and the principal point at the image center."""
    focal = 0.5 * height / math.tan(math.radians(fov_y_deg) / 2.0)
    return torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)) -> torch.Tensor:
    """World-to-camera matrix for a camera at `eye` looking at `target`."""
    eye, target, up = (
        torch.as_tensor(x, dtype=torch.float64) for x in (eye, target, up)
    )
    forward = target - eye
    forward = forward / forward.norm()
    right = torch.linalg.cross(forward, up)
    if right.norm() < 1e-8:
        raise ValueError("up vector is parallel to the viewing direction")
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    R = torch.stack([right, down, forward])
    extrinsics = torch.eye(4, dtype=torch.float64)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = -R @ eye
    return extrinsics


def orbit_cameras(
    n_views: int = 8,
    distance: float = 4.5,
    elevation_range_deg: tuple[float, float] = (-20.0, 60.0),
    width: int = 256,
    height: int = 256,
    fov_y_deg: float = 50.0,
) -> list[Camera]:
    """Cameras on a golden-angle spiral around the origin, all looking at it.

    Elevation sweeps linearly across the range and azimuth advances by the
    golden angle, so no two views coincide even for symmetric shapes like the
    cube.
    """
    K = intrinsics(width, height, fov_y_deg)
    low, high = (math.radians(x) for x in elevation_range_deg)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    cameras = []
    for k in range(n_views):
        elevation = low + (high - low) * (k + 0.5) / n_views
        azimuth = k * golden_angle + math.pi / 8.0  # offset avoids a face-on first view
        eye = distance * torch.tensor(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ]
        )
        cameras.append(Camera(K, look_at(eye), width, height))
    return cameras


def camera_rays(camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray origin and world-space directions scaled so camera-frame z == 1.

    With this scaling the ray parameter t at a hit equals its z-depth.
    """
    v, u = torch.meshgrid(
        torch.arange(camera.height, dtype=torch.float64) + 0.5,
        torch.arange(camera.width, dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1).reshape(-1, 3)
    dirs_cam = pixels @ torch.linalg.inv(camera.K).T
    R = camera.world_to_cam[:3, :3]
    return camera.center, dirs_cam @ R  # row-vector form of R.T @ d


def mesh_intersector(mesh: Mesh, chunk: int = 1024, eps: float = 1e-9) -> Intersector:
    """Moller-Trumbore for a shared ray origin, vectorized as matmuls.

    Rays are processed in chunks; each chunk only tests triangles whose
    bounding cone overlaps the chunk's cone of ray directions. Pass spatially
    coherent rays (render() sends image tiles) for the culling to pay off.
    """
    v0, v1, v2 = mesh.vertices[mesh.faces].unbind(dim=1)
    n0, n1, n2 = mesh.vertex_normals[mesh.faces].unbind(dim=1)
    e1, e2 = v1 - v0, v2 - v0

    def intersect(
        origin: torch.Tensor, dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Bounding cone of each triangle as seen from the origin. Caps narrower
        # than a hemisphere are convex, so the cap through the three vertex
        # directions contains the whole triangle.
        corners = mesh.vertices[mesh.faces] - origin
        corners = corners / corners.norm(dim=-1, keepdim=True)
        face_axis = corners.sum(dim=1)
        face_axis = face_axis / face_axis.norm(dim=-1, keepdim=True)
        face_cos = (corners * face_axis[:, None]).sum(dim=-1).amin(dim=1)
        face_angle = torch.acos(face_cos.clamp(-1.0, 1.0))

        t_out = torch.full((dirs.shape[0],), math.inf, dtype=torch.float64)
        n_out = torch.zeros_like(dirs)
        for start in range(0, dirs.shape[0], chunk):
            d = dirs[start : start + chunk]
            unit = d / d.norm(dim=-1, keepdim=True)
            ray_axis = unit.mean(dim=0)
            ray_axis = ray_axis / ray_axis.norm()
            ray_angle = torch.acos((unit @ ray_axis).clamp(-1.0, 1.0)).max()
            gap = torch.acos((face_axis @ ray_axis).clamp(-1.0, 1.0))
            keep = (gap <= ray_angle + face_angle + 1e-6) | (face_angle > math.pi / 3.0)
            faces = keep.nonzero().squeeze(1)
            if faces.numel() == 0:
                continue

            a, e1_k, e2_k = v0[faces], e1[faces], e2[faces]
            s = origin - a
            q = torch.linalg.cross(s, e1_k)
            det = -(d @ torch.linalg.cross(e1_k, e2_k).T)  # det = -d . (e1 x e2)
            valid = det.abs() > 1e-12
            inv_det = torch.where(valid, 1.0 / torch.where(valid, det, 1.0), 0.0)
            bu = (d @ torch.linalg.cross(e2_k, s).T) * inv_det  # u * det = d . (e2 x s)
            bv = (d @ q.T) * inv_det
            t = (e2_k * q).sum(dim=-1) * inv_det  # t * det = e2 . q
            hit = (
                valid & (bu >= -eps) & (bv >= -eps) & (bu + bv <= 1.0 + eps) & (t > eps)
            )
            t = torch.where(hit, t, math.inf)

            t_min, local = t.min(dim=1)
            rows = torch.arange(d.shape[0])
            bu, bv = bu[rows, local], bv[rows, local]
            bw = 1.0 - bu - bv
            face = faces[local]
            normal = (
                bw[:, None] * n0[face] + bu[:, None] * n1[face] + bv[:, None] * n2[face]
            )
            normal = normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            t_out[start : start + chunk] = t_min
            n_out[start : start + chunk] = torch.where(
                torch.isfinite(t_min)[:, None], normal, 0.0
            )
        return t_out, n_out

    return intersect


def sphere_intersector(radius: float = 1.0) -> Intersector:
    """Analytic ray/sphere intersection (reference for the mesh renderer)."""

    def intersect(
        origin: torch.Tensor, dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = (dirs * dirs).sum(dim=-1)
        b = 2.0 * dirs @ origin
        c = origin @ origin - radius**2
        disc = b * b - 4.0 * a * c
        t = (-b - disc.clamp_min(0.0).sqrt()) / (2.0 * a)
        hit = (disc >= 0) & (t > 0)
        t = torch.where(hit, t, math.inf)
        normal = (origin + torch.where(hit, t, 0.0)[:, None] * dirs) / radius
        return t, torch.where(hit[:, None], normal, 0.0)

    return intersect


def box_intersector(half_extent: float = 1.0) -> Intersector:
    """Analytic ray/axis-aligned-box intersection via the slab method."""

    def intersect(
        origin: torch.Tensor, dirs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / dirs
        t0 = (-half_extent - origin) * inv
        t1 = (half_extent - origin) * inv
        t_near, axis = torch.minimum(t0, t1).max(dim=-1)
        t_far = torch.maximum(t0, t1).min(dim=-1).values
        hit = (t_near <= t_far) & (t_near > 0)
        t = torch.where(hit, t_near, math.inf)
        normal = torch.zeros_like(dirs)
        rows = torch.arange(dirs.shape[0])
        normal[rows, axis] = -torch.sign(dirs[rows, axis])
        return t, torch.where(hit[:, None], normal, 0.0)

    return intersect


def render(
    camera: Camera, intersect: Intersector, tile: int = 32
) -> dict[str, torch.Tensor]:
    """Per-pixel depth, camera-frame normals, and coverage mask for one view."""
    origin, dirs = camera_rays(camera)
    # Group rays into tile x tile blocks so each intersect call sees a narrow cone.
    pixel = torch.arange(camera.height * camera.width)
    tile_id = (pixel // camera.width // tile) * camera.width + (
        pixel % camera.width
    ) // tile
    order = torch.argsort(tile_id, stable=True)
    t = torch.empty(order.shape[0], dtype=torch.float64)
    normals_world = torch.empty_like(dirs)
    for block in order.split(tile * tile):
        t[block], normals_world[block] = intersect(origin, dirs[block])
    mask = torch.isfinite(t)
    R = camera.world_to_cam[:3, :3]
    shape = (camera.height, camera.width)
    return {
        "depth": torch.where(mask, t, 0.0).reshape(shape),
        "normals": (normals_world @ R.T).reshape(*shape, 3),
        "mask": mask.reshape(shape),
    }


def backproject(depth: torch.Tensor, camera: Camera) -> torch.Tensor:
    """Lift a z-depth map to world-space points (H, W, 3)."""
    origin, dirs = camera_rays(camera)
    return (origin + depth.reshape(-1, 1) * dirs).reshape(*depth.shape, 3)
