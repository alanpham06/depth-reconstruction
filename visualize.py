"""Visualize saved sphere and cube point clouds in 3D."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d.axes3d import Axes3D


def _scatter_cloud(
    ax: Axes3D, points: torch.Tensor, normals: torch.Tensor, title: str
) -> None:
    xyz = points.detach().cpu().numpy()
    colors = (normals.detach().cpu().numpy() + 1.0) * 0.5
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=4, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)


def load_shape(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise SystemExit(f"missing {path}. Run point_cloud.py first.")
    return torch.load(path, map_location="cpu", weights_only=True)


def plot_shapes(shapes: dict[str, dict[str, torch.Tensor]]) -> plt.Figure:
    """Draw saved clouds side by side, colored by normals."""
    figure, axes = plt.subplots(
        1, len(shapes), figsize=(6 * len(shapes), 6), subplot_kw={"projection": "3d"}
    )
    if len(shapes) == 1:
        axes = [axes]
    for axis, (name, data) in zip(axes, shapes.items()):
        _scatter_cloud(axis, data["points"], data["normals"], name)
    figure.tight_layout()
    return figure


def main() -> None:
    here = Path(__file__).resolve().parent
    shapes = {
        "sphere": load_shape(here / "sphere.pt"),
        "cube": load_shape(here / "cube.pt"),
    }
    figure = plot_shapes(shapes)

    image_path = Path(__file__).with_name("point_clouds.png")
    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    print(f"saved: {image_path}")
    if plt.get_backend().lower() == "agg":
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
