"""Render ground truth and sparse depth inputs for every procedural primitive.

Writes one file per primitive to data/<name>_gt.pt:

  vertices, faces, vertex_normals   the mesh that was rendered
  K             (3, 3)              intrinsics shared by all views
  world_to_cam  (V, 4, 4)           extrinsics per view (OpenCV convention)
  depth         (V, H, W)           z-depth, 0 where the ray misses
  normals       (V, H, W, 3)        unit camera-frame normals, 0 off the mask
  mask          (V, H, W)           surface coverage
  sparse_depth  (V, H, W)           z-depth of visible sampled surface points, 0 elsewhere
  sparse_mask   (V, H, W)           1-channel binary mask of pixels holding a point

Each view gets a fresh set of --points surface samples (area-weighted, plus
extra density at sharp edges and corners); only unoccluded, in-frame points
reach the sparse map. A preview grid is written to data/<name>_gt.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from geometry import PRIMITIVES
from render import mesh_intersector, orbit_cameras, render
from sampling import sample_surface, sparse_depth_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--size", type=int, default=256, help="image width and height")
    parser.add_argument(
        "--fov", type=float, default=50.0, help="vertical field of view, degrees"
    )
    parser.add_argument(
        "--distance", type=float, default=4.5, help="camera distance from the origin"
    )
    parser.add_argument(
        "--elevation",
        type=float,
        nargs=2,
        default=(-20.0, 60.0),
        metavar=("LOW", "HIGH"),
        help="degrees",
    )
    parser.add_argument(
        "--points", type=int, default=5000, help="surface samples drawn per view"
    )
    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=0.25,
        help="share of samples near sharp edges",
    )
    parser.add_argument(
        "--corner-fraction",
        type=float,
        default=0.05,
        help="share of samples near corners",
    )
    parser.add_argument(
        "--band", type=float, default=0.03, help="width scale of the edge/corner bands"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "data"
    )
    return parser.parse_args()


def save_preview(name: str, data: dict[str, torch.Tensor], path: Path) -> None:
    n_views = data["depth"].shape[0]
    figure, axes = plt.subplots(3, n_views, figsize=(2 * n_views, 6.3), squeeze=False)
    for view in range(n_views):
        depth = data["depth"][view].numpy().copy()
        mask = data["mask"][view].numpy()
        depth[~mask] = float("nan")
        axes[0, view].imshow(depth, cmap="viridis")
        # (N + 1) / 2 is the usual RGB encoding of a normal map.
        axes[1, view].imshow(
            (
                (data["normals"][view] + 1.0) * 0.5 * data["mask"][view, ..., None]
            ).numpy()
        )
        # Same color scale as the dense map so the two are directly comparable.
        sparse = data["sparse_depth"][view].numpy().copy()
        sparse[~data["sparse_mask"][view].numpy()] = float("nan")
        axes[2, view].imshow(
            sparse,
            cmap="viridis",
            vmin=float(depth[mask].min()),
            vmax=float(depth[mask].max()),
        )
        for axis in axes[:, view]:
            axis.set_axis_off()
        axes[0, view].set_title(f"view {view}", fontsize=8)
    figure.suptitle(f"{name}: depth, normals, sparse depth")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cameras = orbit_cameras(
        n_views=args.views,
        distance=args.distance,
        elevation_range_deg=tuple(args.elevation),
        width=args.size,
        height=args.size,
        fov_y_deg=args.fov,
    )
    for name, build in PRIMITIVES.items():
        mesh = build()
        intersect = mesh_intersector(mesh)
        views = [render(camera, intersect) for camera in cameras]
        generator = torch.Generator().manual_seed(args.seed)
        sparse = [
            sparse_depth_map(
                sample_surface(
                    mesh,
                    args.points,
                    edge_fraction=args.edge_fraction,
                    corner_fraction=args.corner_fraction,
                    band=args.band,
                    generator=generator,
                ).points,
                camera,
                intersect,
            )
            for camera in cameras
        ]
        data = {
            "vertices": mesh.vertices.float(),
            "faces": mesh.faces,
            "vertex_normals": mesh.vertex_normals.float(),
            "K": cameras[0].K.float(),
            "world_to_cam": torch.stack(
                [camera.world_to_cam for camera in cameras]
            ).float(),
            "depth": torch.stack([view["depth"] for view in views]).float(),
            "normals": torch.stack([view["normals"] for view in views]).float(),
            "mask": torch.stack([view["mask"] for view in views]),
            "sparse_depth": torch.stack([depth for depth, _ in sparse]).float(),
            "sparse_mask": torch.stack([mask for _, mask in sparse]),
        }
        torch.save(data, args.out / f"{name}_gt.pt")
        save_preview(name, data, args.out / f"{name}_gt.png")
        coverage = data["mask"].float().mean().item()
        sparsity = data["sparse_mask"].float().mean().item()
        on_object = (data["sparse_mask"].sum() / data["mask"].sum()).item()
        print(
            f"{name}: verts={mesh.vertices.shape[0]} faces={mesh.faces.shape[0]} "
            f"depth={tuple(data['depth'].shape)} coverage={coverage:.1%} "
            f"sparse={sparsity:.2%} of image, {on_object:.1%} of object -> {args.out / f'{name}_gt.pt'}"
        )


if __name__ == "__main__":
    main()
