"""Generate procedural 3D ground truth with PyTorch only.

The saved tensors contain surface points and normals that can be sampled by a
downstream reconstruction model. Coordinates are centered at the origin and
fit inside [-1, 1] in every axis.
"""

from pathlib import Path

import torch


def sphere_point_cloud(
	n_lat: int = 34,
	n_lon: int = 128,
	radius: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return UV-sphere vertices and outward normals as a point cloud.

	Latitude rings run from the north pole (phi = 0) to the south pole
	(phi = pi). Longitude samples the full circle without repeating 0 and 2pi.
	Poles are stored once so they are not duplicated n_lon times.
	"""
	if n_lat < 2:
		raise ValueError("n_lat must be >= 2")
	if n_lon < 3:
		raise ValueError("n_lon must be >= 3")
	if radius <= 0:
		raise ValueError("radius must be > 0")

	north = torch.tensor([[0.0, 0.0, 1.0]])
	south = torch.tensor([[0.0, 0.0, -1.0]])

	if n_lat == 2:
		unit = torch.cat([north, south], dim=0)
	else:
		phi = torch.linspace(0.0, torch.pi, n_lat)[1:-1]
		theta = torch.linspace(0.0, 2.0 * torch.pi, n_lon + 1)[:-1]
		phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="ij")

		# Convert spherical coordinates to Cartesian coordinates 
		sin_phi = torch.sin(phi_grid)
		x = sin_phi * torch.cos(theta_grid)
		y = sin_phi * torch.sin(theta_grid)
		z = torch.cos(phi_grid)
		
		body = torch.stack((x, y, z), dim=-1).reshape(-1, 3)
		unit = torch.cat([north, body, south], dim=0)

	points = radius * unit
	return points, unit.clone()


def cube_point_cloud(
	n_side: int = 27,
	half_extent: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Return a cube surface point cloud from six regular face grids.

	Each face is an n_side x n_side lattice including its boundary. Shared
	edge and corner vertices are then uniqued so the cloud is not denser
	along the 12 edges.
	"""
	if n_side < 2:
		raise ValueError("n_side must be >= 2")
	if half_extent <= 0:
		raise ValueError("half_extent must be > 0")

	axis_values = torch.linspace(-half_extent, half_extent, n_side)
	grid_u, grid_v = torch.meshgrid(axis_values, axis_values, indexing="ij")
	face_u = grid_u.reshape(-1)
	face_v = grid_v.reshape(-1)

	faces = []
	for axis in range(3):
		for sign in (-1.0, 1.0):
			points = torch.zeros((face_u.shape[0], 3))
			free_axes = [index for index in range(3) if index != axis]
			points[:, free_axes[0]] = face_u
			points[:, free_axes[1]] = face_v
			points[:, axis] = sign * half_extent
			faces.append(points)

	points = torch.unique(torch.cat(faces, dim=0), dim=0)

	face_axis = points.abs().argmax(dim=-1)
	normals = torch.zeros_like(points)
	row = torch.arange(points.shape[0])
	normals[row, face_axis] = torch.sign(points[row, face_axis])
	return points, normals


def sphere_sdf(points: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
	"""Signed distance to a sphere: negative inside, zero on the surface."""
	return points.norm(dim=-1) - radius


def cube_sdf(points: torch.Tensor, half_extent: float = 1.0) -> torch.Tensor:
	"""Signed distance to an axis-aligned cube."""
	offset = points.abs() - half_extent
	outside_distance = torch.clamp(offset, min=0.0).norm(dim=-1)
	inside_distance = torch.clamp(offset.amax(dim=-1), max=0.0)
	return outside_distance + inside_distance


def build_sphere() -> dict[str, torch.Tensor]:
	"""Create reproducible sphere surface data."""
	points, normals = sphere_point_cloud()
	return {
		"points": points,
		"normals": normals,
		"sdf": sphere_sdf(points),
	}


def build_cube() -> dict[str, torch.Tensor]:
	"""Create reproducible cube surface data."""
	points, normals = cube_point_cloud()
	return {
		"points": points,
		"normals": normals,
		"sdf": cube_sdf(points),
	}


def _save_shape(name: str, data: dict[str, torch.Tensor], output_path: Path) -> None:
	torch.save(data, output_path)
	print(
		f"{name}: points={tuple(data['points'].shape)}, "
		f"normals={tuple(data['normals'].shape)}, "
		f"max_abs_sdf={data['sdf'].abs().max().item():.2e}"
	)
	print(f"saved: {output_path}")


def main() -> None:
	here = Path(__file__).resolve().parent
	_save_shape("sphere", build_sphere(), here / "sphere.pt")
	_save_shape("cube", build_cube(), here / "cube.pt")


if __name__ == "__main__":
	main()
