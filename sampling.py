"""Surface point sampling and sparse depth projection.

Sampling mixes two strategies:

  * area-weighted uniform sampling: a face is chosen with probability
    proportional to its area, then a uniform barycentric point inside it, so
    point density is constant per unit surface area;
  * feature sampling: extra points concentrated in a thin band around sharp
    edges (dihedral angle above a threshold) and around corners where three or
    more sharp edges meet.

Sharp features are detected from the mesh itself. Smooth meshes like the
spheres have none, so all of their samples come from the uniform sampler; the
cube gets extra density along its 12 edges and 8 corners.

Every sample lies exactly on a mesh triangle, so projected depths agree with the
rendered ground truth.
"""

import math
from dataclasses import dataclass

import torch

from geometry import Mesh
from render import Camera, Intersector


@dataclass
class SurfaceSamples:
    points: torch.Tensor  # (N, 3) world coordinates
    normals: torch.Tensor  # (N, 3) interpolated unit normals
    faces: torch.Tensor  # (N,) index of the triangle each point lies on
    kind: torch.Tensor  # (N,) 0 = uniform, 1 = edge band, 2 = corner band


@dataclass
class SharpFeatures:
    # Half-edge (face f, slot i) is the triangle edge faces[f, i] -> faces[f, (i + 1) % 3].
    edge_faces: torch.Tensor  # (E,) face of each half-edge on a sharp edge
    edge_slots: torch.Tensor  # (E,) slot of each such half-edge
    # Wedge (face f, slot i) is the corner of triangle f at vertex faces[f, i].
    corner_faces: torch.Tensor  # (C,) faces touching a corner vertex
    corner_slots: torch.Tensor  # (C,) slot of the corner vertex in that face
    n_corners: int


def triangle_areas(mesh: Mesh) -> torch.Tensor:
    v0, v1, v2 = mesh.vertices[mesh.faces].unbind(dim=1)
    return 0.5 * torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=-1)


def find_sharp_features(mesh: Mesh, angle_deg: float = 30.0) -> SharpFeatures:
    """Find edges whose adjacent face normals differ by more than angle_deg.

    Vertices are welded by position first, so meshes built from separate
    patches (like the six cube grids) are handled the same as welded ones.
    Edges with only one adjacent face are open boundaries and count as sharp.
    """
    _, welded = torch.unique((mesh.vertices / 1e-9).round(), dim=0, return_inverse=True)
    faces = welded[mesh.faces]
    n_faces = faces.shape[0]

    half_edges = torch.stack([faces, faces.roll(-1, dims=1)], dim=-1).reshape(-1, 2)
    keys, edge_id, counts = torch.unique(
        half_edges.sort(dim=-1).values, dim=0, return_inverse=True, return_counts=True
    )
    face_of = torch.arange(n_faces).repeat_interleave(3)

    # Pair the two half-edges of every manifold edge and compare their face normals.
    order = torch.argsort(edge_id, stable=True)
    sorted_ids = edge_id[order]
    first = torch.ones_like(sorted_ids, dtype=torch.bool)
    first[1:] = sorted_ids[1:] != sorted_ids[:-1]
    start = order[first]  # first half-edge of each undirected edge
    pair_ok = counts == 2
    second = torch.full_like(start, -1)
    second[pair_ok] = order[torch.nonzero(first).squeeze(1)[pair_ok] + 1]

    face_normals = mesh.face_normals()
    cos = torch.ones(keys.shape[0], dtype=torch.float64)
    cos[pair_ok] = (
        face_normals[face_of[start[pair_ok]]] * face_normals[face_of[second[pair_ok]]]
    ).sum(-1)
    sharp_edge = (cos < math.cos(math.radians(angle_deg))) | (counts == 1)

    sharp_half = sharp_edge[edge_id]
    edge_faces = face_of[sharp_half]
    edge_slots = torch.arange(3).repeat(n_faces)[sharp_half]

    # Corners: welded vertices where three or more sharp edges meet.
    sharp_keys = keys[sharp_edge]
    valence = torch.bincount(sharp_keys.reshape(-1), minlength=int(welded.max()) + 1)
    is_corner = valence >= 3
    corner_wedge = is_corner[faces]  # (F, 3)
    corner_faces, corner_slots = corner_wedge.nonzero(as_tuple=True)
    return SharpFeatures(
        edge_faces, edge_slots, corner_faces, corner_slots, int(is_corner.sum())
    )


def _interpolate_normals(
    mesh: Mesh, faces: torch.Tensor, bary: torch.Tensor
) -> torch.Tensor:
    normals = (mesh.vertex_normals[mesh.faces[faces]] * bary[..., None]).sum(dim=1)
    return normals / normals.norm(dim=-1, keepdim=True)


def _samples(
    mesh: Mesh, faces: torch.Tensor, bary: torch.Tensor, kind: int
) -> SurfaceSamples:
    points = (mesh.vertices[mesh.faces[faces]] * bary[..., None]).sum(dim=1)
    return SurfaceSamples(
        points,
        _interpolate_normals(mesh, faces, bary),
        faces,
        torch.full_like(faces, kind),
    )


def sample_uniform(
    mesh: Mesh, n: int, generator: torch.Generator | None = None
) -> SurfaceSamples:
    """Area-weighted face choice plus uniform barycentrics (sqrt trick)."""
    faces = torch.multinomial(
        triangle_areas(mesh), n, replacement=True, generator=generator
    )
    r1, r2 = torch.rand(2, n, dtype=torch.float64, generator=generator)
    s = r1.sqrt()
    bary = torch.stack([1.0 - s, s * (1.0 - r2), s * r2], dim=-1)
    return _samples(mesh, faces, bary, kind=0)


def _band_offset(
    n: int, band: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Distances from a feature, half-normal with scale `band`."""
    return (torch.randn(n, dtype=torch.float64, generator=generator) * band).abs()


def sample_edges(
    mesh: Mesh,
    features: SharpFeatures,
    n: int,
    band: float,
    generator: torch.Generator | None = None,
) -> SurfaceSamples:
    """Points within ~band of sharp edges, uniform along each edge's length.

    A half-edge is picked with probability proportional to its length (both
    sides of an edge are half-edges, so points spread onto both faces). The
    point slides from a uniform spot on the edge toward the opposite vertex by
    the band distance, which keeps it inside the triangle.
    """
    f, i = features.edge_faces, features.edge_slots
    tri = mesh.vertices[mesh.faces[f]]  # (E, 3, 3)
    rows = torch.arange(f.shape[0])
    a, b = tri[rows, i], tri[rows, (i + 1) % 3]
    length = (b - a).norm(dim=-1)
    height = 2.0 * triangle_areas(mesh)[f] / length  # distance from c to the edge

    pick = torch.multinomial(length, n, replacement=True, generator=generator)
    along = torch.rand(n, dtype=torch.float64, generator=generator)
    toward = (_band_offset(n, band, generator) / height[pick]).clamp(max=1.0)

    # p = (1 - w) * (a + s (b - a)) + w * c, written as barycentrics over the face slots.
    bary = torch.zeros(n, 3, dtype=torch.float64)
    slot = i[pick]
    sample_rows = torch.arange(n)
    bary[sample_rows, slot] = (1.0 - toward) * (1.0 - along)
    bary[sample_rows, (slot + 1) % 3] = (1.0 - toward) * along
    bary[sample_rows, (slot + 2) % 3] = toward
    return _samples(mesh, f[pick], bary, kind=1)


def sample_corners(
    mesh: Mesh,
    features: SharpFeatures,
    n: int,
    band: float,
    generator: torch.Generator | None = None,
) -> SurfaceSamples:
    """Points within ~band of corner vertices, spread over the incident faces.

    A wedge (a triangle touching the corner) is picked with probability
    proportional to its interior angle, so points fan out evenly around the
    corner. The point then moves from the corner into that triangle.
    """
    f, i = features.corner_faces, features.corner_slots
    tri = mesh.vertices[mesh.faces[f]]
    rows = torch.arange(f.shape[0])
    v, b, c = tri[rows, i], tri[rows, (i + 1) % 3], tri[rows, (i + 2) % 3]
    eb, ec = b - v, c - v
    angle = torch.acos(
        ((eb * ec).sum(-1) / (eb.norm(dim=-1) * ec.norm(dim=-1))).clamp(-1.0, 1.0)
    )

    pick = torch.multinomial(angle, n, replacement=True, generator=generator)
    mix = torch.rand(n, dtype=torch.float64, generator=generator)
    direction_length = (mix[:, None] * eb[pick] + (1.0 - mix[:, None]) * ec[pick]).norm(
        dim=-1
    )
    reach = (_band_offset(n, band, generator) / direction_length).clamp(max=1.0)

    bary = torch.zeros(n, 3, dtype=torch.float64)
    slot = i[pick]
    sample_rows = torch.arange(n)
    bary[sample_rows, slot] = 1.0 - reach
    bary[sample_rows, (slot + 1) % 3] = reach * mix
    bary[sample_rows, (slot + 2) % 3] = reach * (1.0 - mix)
    return _samples(mesh, f[pick], bary, kind=2)


def sample_surface(
    mesh: Mesh,
    n: int,
    edge_fraction: float = 0.25,
    corner_fraction: float = 0.05,
    band: float = 0.03,
    sharp_angle_deg: float = 30.0,
    generator: torch.Generator | None = None,
) -> SurfaceSamples:
    """Draw n surface points: uniform by area, plus extra density at sharp features.

    edge_fraction / corner_fraction of the budget goes to the edge and corner
    bands. Meshes without such features spend the whole budget uniformly.
    """
    features = find_sharp_features(mesh, sharp_angle_deg)
    n_edge = round(n * edge_fraction) if features.edge_faces.numel() else 0
    n_corner = round(n * corner_fraction) if features.corner_faces.numel() else 0
    parts = [sample_uniform(mesh, n - n_edge - n_corner, generator)]
    if n_edge:
        parts.append(sample_edges(mesh, features, n_edge, band, generator))
    if n_corner:
        parts.append(sample_corners(mesh, features, n_corner, band, generator))
    return SurfaceSamples(
        points=torch.cat([p.points for p in parts]),
        normals=torch.cat([p.normals for p in parts]),
        faces=torch.cat([p.faces for p in parts]),
        kind=torch.cat([p.kind for p in parts]),
    )


def project_points(
    points: torch.Tensor, camera: Camera
) -> tuple[torch.Tensor, torch.Tensor]:
    """World points -> (continuous pixel coords (N, 2), camera-frame z (N,))."""
    R, t = camera.world_to_cam[:3, :3], camera.world_to_cam[:3, 3]
    cam = points @ R.T + t
    pixels = cam @ camera.K.T
    return pixels[:, :2] / pixels[:, 2:3], cam[:, 2]


def visible_points(
    points: torch.Tensor,
    camera: Camera,
    intersect: Intersector,
    rel_tol: float = 1e-6,
    tile: int = 32,
) -> torch.Tensor:
    """True for points in frame and not occluded by the surface.

    The ray from the camera center through each point is cast against the
    surface; the point is visible when the first hit is the point itself.
    """
    uv, z = project_points(points, camera)
    in_frame = (
        (z > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < camera.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < camera.height)
    )
    candidates = in_frame.nonzero().squeeze(1)
    visible = torch.zeros(points.shape[0], dtype=torch.bool)
    if candidates.numel() == 0:
        return visible

    # Group rays by image tile so the intersector's cone culling stays effective.
    cand_uv = uv[candidates].floor().long()
    tile_id = (cand_uv[:, 1] // tile) * camera.width + cand_uv[:, 0] // tile
    candidates = candidates[torch.argsort(tile_id, stable=True)]
    tile_id = tile_id.sort(stable=True).values
    origin = camera.center
    dirs = (points[candidates] - origin) / z[candidates, None]  # scaled so t == z-depth
    _, counts = torch.unique_consecutive(tile_id, return_counts=True)
    for block, block_dirs in zip(
        candidates.split(counts.tolist()), dirs.split(counts.tolist())
    ):
        t_hit, _ = intersect(origin, block_dirs)
        visible[block] = t_hit >= z[block] * (1.0 - rel_tol)
    return visible


def sparse_depth_map(
    points: torch.Tensor, camera: Camera, intersect: Intersector
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize visible sample points into a sparse z-depth map and binary mask.

    Each visible point lands in the pixel containing its projection and writes
    its exact z-depth; if several land in one pixel the nearest wins. Pixels
    without a point are 0 in both outputs.
    """
    visible = visible_points(points, camera, intersect)
    uv, z = project_points(points[visible], camera)
    pixel = uv[:, 1].floor().long() * camera.width + uv[:, 0].floor().long()
    flat = torch.full((camera.height * camera.width,), math.inf, dtype=torch.float64)
    flat = flat.scatter_reduce(0, pixel, z, reduce="amin")
    mask = torch.isfinite(flat)
    depth = torch.where(mask, flat, 0.0)
    shape = (camera.height, camera.width)
    return depth.reshape(shape), mask.reshape(shape)
