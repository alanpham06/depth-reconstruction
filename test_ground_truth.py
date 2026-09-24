"""Checks that the procedural meshes and ground-truth renders meet spec.

Run with `python test_ground_truth.py` (or pytest, if installed).
"""

import math

import torch

from geometry import Mesh, cube_mesh, icosphere_mesh, uv_sphere_mesh
from render import (
	backproject,
	box_intersector,
	camera_rays,
	intrinsics,
	look_at,
	mesh_intersector,
	orbit_cameras,
	render,
	sphere_intersector,
)

CAMERAS = orbit_cameras(n_views=4, width=128, height=128)


def _euler_characteristic(mesh: Mesh) -> int:
	edges = torch.cat([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
	n_edges = torch.unique(edges.sort(dim=-1).values, dim=0).shape[0]
	return mesh.vertices.shape[0] - n_edges + mesh.faces.shape[0]


def _assert_closed_manifold(mesh: Mesh) -> None:
	edges = torch.cat([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
	_, counts = torch.unique(edges.sort(dim=-1).values, dim=0, return_counts=True)
	assert (counts == 2).all(), "every edge must be shared by exactly two faces"
	# Directed edges must each appear once: consistent winding across the surface.
	assert torch.unique(edges, dim=0).shape[0] == edges.shape[0]
	assert _euler_characteristic(mesh) == 2


def _weld(mesh: Mesh) -> Mesh:
	vertices, inverse = torch.unique(mesh.vertices, dim=0, return_inverse=True)
	return Mesh(vertices, inverse[mesh.faces], torch.zeros_like(vertices))


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	return torch.rad2deg(torch.acos((a * b).sum(-1).clamp(-1.0, 1.0)))


# --- Geometry engine ---------------------------------------------------------


def test_spheres_are_closed_outward_and_on_surface():
	for mesh in (uv_sphere_mesh(), icosphere_mesh()):
		_assert_closed_manifold(mesh)
		assert torch.allclose(mesh.vertices.norm(dim=-1), torch.ones(1, dtype=torch.float64))
		assert torch.allclose(mesh.vertex_normals, mesh.vertices)
		centroids = mesh.vertices[mesh.faces].mean(dim=1)
		assert ((mesh.face_normals() * centroids).sum(-1) > 0).all(), "faces must wind outward"
		assert abs(mesh.signed_volume() - 4.0 / 3.0 * math.pi) < 0.02


def test_icosphere_subdivision_counts():
	for level in range(4):
		mesh = icosphere_mesh(subdivisions=level)
		assert mesh.faces.shape[0] == 20 * 4**level
		assert mesh.vertices.shape[0] == 10 * 4**level + 2


def test_cube_is_six_plane_grids():
	n_side = 27
	mesh = cube_mesh(n_side=n_side)
	assert mesh.vertices.shape[0] == 6 * n_side**2
	assert mesh.faces.shape[0] == 6 * 2 * (n_side - 1) ** 2
	assert torch.allclose(mesh.vertices.abs().amax(dim=-1), torch.ones(1, dtype=torch.float64))
	assert torch.allclose(mesh.face_normals(), mesh.vertex_normals[mesh.faces[:, 0]])
	assert abs(mesh.signed_volume() - 8.0) < 1e-9
	# Faces are separate grids with sharp edges; welding coincident vertices closes the surface.
	_assert_closed_manifold(_weld(mesh))


# --- Cameras -----------------------------------------------------------------


def test_intrinsics_and_extrinsics_project_consistently():
	K = intrinsics(640, 480, fov_y_deg=60.0)
	assert math.isclose(K[1, 1].item(), 240.0 / math.tan(math.radians(30.0)))
	extrinsics = look_at(eye=(3.0, 0.0, 0.0))
	R = extrinsics[:3, :3]
	assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64))
	assert math.isclose(torch.linalg.det(R).item(), 1.0)
	# The look-at target lands on the principal point at depth 3.
	target_cam = extrinsics @ torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
	assert torch.allclose(target_cam[:3], torch.tensor([0.0, 0.0, 3.0], dtype=torch.float64))
	# World +z (up) projects above the image center, i.e. smaller v.
	up_cam = (extrinsics @ torch.tensor([0.0, 0.0, 0.5, 1.0], dtype=torch.float64))[:3]
	pixel = K @ up_cam
	assert pixel[1] / pixel[2] < K[1, 2]


def test_ray_parameter_is_z_depth():
	camera = CAMERAS[0]
	_, dirs = camera_rays(camera)
	dirs_cam = dirs @ camera.world_to_cam[:3, :3].T
	assert torch.allclose(dirs_cam[:, 2], torch.ones(1, dtype=torch.float64))


# --- Ground-truth rendering --------------------------------------------------


def _check_render_invariants(view: dict[str, torch.Tensor], camera, grazing_tol_deg: float = 0.0) -> None:
	mask, normals = view["mask"], view["normals"]
	assert mask.any()
	assert (view["depth"][~mask] == 0).all() and (normals[~mask] == 0).all()
	assert torch.allclose(normals[mask].norm(dim=-1), torch.ones(1, dtype=torch.float64))
	# Visible normals face the camera: n . ray < 0 in camera frame. Interpolated
	# normals on a faceted silhouette may tip a degree or so past perpendicular.
	_, dirs = camera_rays(camera)
	dirs_cam = (dirs @ camera.world_to_cam[:3, :3].T).reshape(*mask.shape, 3)
	dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True)
	assert ((normals * dirs_cam).sum(-1)[mask] < math.sin(math.radians(grazing_tol_deg))).all()


def test_cube_mesh_render_matches_analytic_box():
	intersect_mesh = mesh_intersector(cube_mesh())
	intersect_exact = box_intersector()
	for camera in CAMERAS:
		mesh_view, exact_view = render(camera, intersect_mesh), render(camera, intersect_exact)
		_check_render_invariants(mesh_view, camera)
		assert (mesh_view["mask"] == exact_view["mask"]).all()
		both = mesh_view["mask"]
		assert (mesh_view["depth"] - exact_view["depth"])[both].abs().max() < 1e-9
		assert torch.allclose(mesh_view["normals"], exact_view["normals"], atol=1e-9)
		points = backproject(mesh_view["depth"], camera)[both]
		assert torch.allclose(points.abs().amax(dim=-1), torch.ones(1, dtype=torch.float64), atol=1e-9)


def test_sphere_mesh_render_matches_analytic_sphere():
	intersect_exact = sphere_intersector()
	for mesh in (uv_sphere_mesh(), icosphere_mesh()):
		intersect_mesh = mesh_intersector(mesh)
		for camera in CAMERAS:
			mesh_view, exact_view = render(camera, intersect_mesh), render(camera, intersect_exact)
			_check_render_invariants(mesh_view, camera, grazing_tol_deg=2.0)
			both = mesh_view["mask"] & exact_view["mask"]
			disagree = (mesh_view["mask"] != exact_view["mask"]).float().mean()
			assert disagree < 0.005, "only silhouette pixels may differ"
			# Hits lie on the inscribed facets: radial error is bounded by the facet
			# sagitta (<0.2% of r). Depth error along the ray grows only at grazing rays.
			points = backproject(mesh_view["depth"], camera)[mesh_view["mask"]]
			radius = points.norm(dim=-1)
			assert (radius <= 1.0 + 1e-9).all() and (radius > 0.998).all()
			assert (mesh_view["depth"] - exact_view["depth"])[both].abs().median() < 3e-3
			# Interpolated normals equal the true sphere normal through the hit point.
			true_normals = (points / radius[:, None]) @ camera.world_to_cam[:3, :3].T
			assert _angle_deg(mesh_view["normals"][mesh_view["mask"]], true_normals).max() < 1e-4
			angles = _angle_deg(mesh_view["normals"][both], exact_view["normals"][both])
			assert angles.mean() < 0.2


def test_sphere_normals_are_continuous():
	"""Pixel-to-pixel normal change tracks the analytic sphere, with no facet jumps."""
	camera = orbit_cameras(n_views=1, width=256, height=256)[0]
	mesh_view = render(camera, mesh_intersector(icosphere_mesh()))
	exact_view = render(camera, sphere_intersector())

	def jumps(view):
		normals = view["normals"]
		return _angle_deg(normals[1:, 1:], normals[1:, :-1]), _angle_deg(normals[1:, 1:], normals[:-1, 1:])

	# Stay off the silhouette, where the true normal turns arbitrarily fast per pixel.
	facing = mesh_view["mask"] & exact_view["mask"] & (exact_view["normals"][..., 2] < -0.5)
	interior = facing[1:, 1:] & facing[:-1, 1:] & facing[1:, :-1]
	for mesh_jump, exact_jump in zip(jumps(mesh_view), jumps(exact_view)):
		# Flat shading would give 0 deg inside facets and ~5 deg spikes at edges.
		assert (mesh_jump - exact_jump)[interior].abs().max() < 0.5


if __name__ == "__main__":
	tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_")]
	for name, fn in tests:
		fn()
		print(f"ok  {name}")
	print(f"{len(tests)} passed")
