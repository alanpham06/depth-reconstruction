"""Procedural triangle meshes for standard 3D primitives, built with PyTorch.

Every mesh is centered at the origin, has counter-clockwise (outward) winding,
and carries per-vertex normals so a renderer can interpolate them into a
continuous normal field.
"""

from dataclasses import dataclass

import torch

from point_cloud import sphere_point_cloud


@dataclass
class Mesh:
    vertices: torch.Tensor  # (V, 3) float64
    faces: torch.Tensor  # (F, 3) int64, CCW when viewed from outside
    vertex_normals: torch.Tensor  # (V, 3) float64, unit length

    def face_normals(self) -> torch.Tensor:
        v0, v1, v2 = self.vertices[self.faces].unbind(dim=1)
        normals = torch.linalg.cross(v1 - v0, v2 - v0)
        return normals / normals.norm(dim=-1, keepdim=True)

    def signed_volume(self) -> float:
        """Positive for a closed mesh with outward winding."""
        v0, v1, v2 = self.vertices[self.faces].unbind(dim=1)
        return (v0 * torch.linalg.cross(v1, v2)).sum().item() / 6.0


def uv_sphere_mesh(n_lat: int = 34, n_lon: int = 128, radius: float = 1.0) -> Mesh:
    """Latitude/longitude sphere triangulated from the point_cloud.py vertices.

    Vertex layout: north pole, (n_lat - 2) rings of n_lon vertices, south pole.
    """
    if n_lat < 3:
        raise ValueError("n_lat must be >= 3 for a triangulated sphere")
    points, normals = sphere_point_cloud(n_lat=n_lat, n_lon=n_lon, radius=radius)
    n_rings = n_lat - 2
    north, south = 0, points.shape[0] - 1

    def ring(i: int) -> torch.Tensor:
        return 1 + i * n_lon + torch.arange(n_lon)

    faces = []
    first = ring(0)
    faces.append(
        torch.stack([torch.full_like(first, north), first, first.roll(-1)], dim=-1)
    )
    for i in range(n_rings - 1):
        a, c = ring(i), ring(i + 1)
        b, d = a.roll(-1), c.roll(-1)
        faces.append(torch.stack([a, c, d], dim=-1))
        faces.append(torch.stack([a, d, b], dim=-1))
    last = ring(n_rings - 1)
    faces.append(
        torch.stack([torch.full_like(last, south), last.roll(-1), last], dim=-1)
    )

    return Mesh(points.double(), torch.cat(faces, dim=0), normals.double())


def icosphere_mesh(subdivisions: int = 4, radius: float = 1.0) -> Mesh:
    """Icosahedron refined by midpoint subdivision and projected onto the sphere."""
    if subdivisions < 0:
        raise ValueError("subdivisions must be >= 0")
    phi = (1.0 + 5.0**0.5) / 2.0
    vertices = torch.tensor(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=torch.float64,
    )
    faces = torch.tensor(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )
    vertices = vertices / vertices.norm(dim=-1, keepdim=True)

    for _ in range(subdivisions):
        # Edges (v0,v1), (v1,v2), (v2,v0) of every face; shared edges get one midpoint.
        edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
        unique_edges, inverse = torch.unique(
            edges.sort(dim=-1).values, dim=0, return_inverse=True
        )
        midpoints = vertices[unique_edges].mean(dim=1)
        midpoints = midpoints / midpoints.norm(dim=-1, keepdim=True)
        mid = (inverse + vertices.shape[0]).reshape(
            3, -1
        )  # mid[k] = midpoint of edge k per face
        m01, m12, m20 = mid
        v0, v1, v2 = faces.unbind(dim=-1)
        faces = torch.cat(
            [
                torch.stack([v0, m01, m20], dim=-1),
                torch.stack([v1, m12, m01], dim=-1),
                torch.stack([v2, m20, m12], dim=-1),
                torch.stack([m01, m12, m20], dim=-1),
            ],
            dim=0,
        )
        vertices = torch.cat([vertices, midpoints], dim=0)

    return Mesh(radius * vertices, faces, vertices.clone())


def cube_mesh(n_side: int = 27, half_extent: float = 1.0) -> Mesh:
    """Axis-aligned cube assembled from six independent n_side x n_side plane grids.

    Vertices are not welded across faces so each face keeps its exact normal and
    the 12 edges stay sharp. Shared edge coordinates are bit-identical, so the
    surface has no cracks.
    """
    if n_side < 2:
        raise ValueError("n_side must be >= 2")
    values = torch.linspace(-half_extent, half_extent, n_side, dtype=torch.float64)
    grid_u, grid_v = torch.meshgrid(values, values, indexing="ij")

    # Two triangles per grid cell; vertex index = i * n_side + j.
    i, j = torch.meshgrid(
        torch.arange(n_side - 1), torch.arange(n_side - 1), indexing="ij"
    )
    p00 = (i * n_side + j).reshape(-1)
    p10, p01, p11 = p00 + n_side, p00 + 1, p00 + n_side + 1
    cell_faces = torch.cat(
        [torch.stack([p00, p10, p11], -1), torch.stack([p00, p11, p01], -1)]
    )

    vertices, faces, normals = [], [], []
    for axis in range(3):
        free_a, free_b = [index for index in range(3) if index != axis]
        for sign in (-1.0, 1.0):
            points = torch.zeros((n_side * n_side, 3), dtype=torch.float64)
            points[:, free_a] = grid_u.reshape(-1)
            points[:, free_b] = grid_v.reshape(-1)
            points[:, axis] = sign * half_extent
            normal = torch.zeros(3, dtype=torch.float64)
            normal[axis] = sign

            # cell_faces winds along +u then +v, i.e. around e_a x e_b; flip if inward.
            e_a, e_b = torch.eye(3, dtype=torch.float64)[[free_a, free_b]]
            face = (
                cell_faces
                if torch.dot(torch.linalg.cross(e_a, e_b), normal) > 0
                else cell_faces.flip(-1)
            )

            faces.append(face + sum(v.shape[0] for v in vertices))
            vertices.append(points)
            normals.append(normal.expand_as(points))

    return Mesh(torch.cat(vertices), torch.cat(faces), torch.cat(normals))


PRIMITIVES = {
    "sphere_uv": uv_sphere_mesh,
    "sphere_ico": icosphere_mesh,
    "cube": cube_mesh,
}
