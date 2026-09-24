"""Interactive Plotly viewer for saved point-cloud .pt files."""

import argparse
import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


def load_shape(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise SystemExit(f"missing {path}. Run point_cloud.py first.")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if "points" not in data or "normals" not in data:
        raise SystemExit(f"{path} must contain 'points' and 'normals' tensors")
    return data


def _marker_colors(normals: np.ndarray) -> list[str]:
    rgb = np.clip((normals + 1.0) * 0.5, 0.0, 1.0) * 255.0
    return [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in rgb]


def _scatter3d(name: str, data: dict[str, torch.Tensor]) -> go.Scatter3d:
    points = data["points"].detach().cpu().numpy()
    normals = data["normals"].detach().cpu().numpy()
    return go.Scatter3d(
        name=name,
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker={"size": 2, "color": _marker_colors(normals)},
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra>"
        + name
        + "</extra>",
    )


def build_figure(shapes: dict[str, dict[str, torch.Tensor]]) -> go.Figure:
    names = list(shapes)
    figure = make_subplots(
        rows=1,
        cols=len(names),
        specs=[[{"type": "scene"}] * len(names)],
        subplot_titles=names,
    )
    for index, name in enumerate(names, start=1):
        figure.add_trace(_scatter3d(name, shapes[name]), row=1, col=index)

    figure.update_scenes(
        aspectmode="cube",
        xaxis={"range": [-1.1, 1.1], "title": "x"},
        yaxis={"range": [-1.1, 1.1], "title": "y"},
        zaxis={"range": [-1.1, 1.1], "title": "z"},
    )
    figure.update_layout(
        title="Drag to rotate, scroll to zoom, right-drag to pan",
        showlegend=False,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        height=700,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive point-cloud viewer")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional .pt files. Defaults to sphere.pt and cube.pt",
    )
    return parser.parse_args()


def _windows_temp_copy(path: Path) -> str:
    """Copy to a C:\\ temp path so Windows does not see a UNC/WSL cwd."""
    user = os.environ.get("USER", "temp")
    dest_dir = Path(f"/mnt/c/Users/{user}/AppData/Local/Temp")
    if not dest_dir.is_dir():
        dest_dir = Path("/mnt/c/Windows/Temp")
    dest = dest_dir / "depth-reconstruction-viewer.html"
    shutil.copy2(path, dest)
    return str(dest).replace("/mnt/c", "C:", 1).replace("/", "\\")


def _spawn_detached(command: list[str]) -> None:
    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        cwd="/mnt/c/Windows",
    )


def open_html(path: Path) -> str:
    """Open HTML in the default app. On WSL, hand a C:\\ copy to Windows."""
    if os.environ.get("WSL_DISTRO_NAME") and shutil.which("explorer.exe"):
        win_path = _windows_temp_copy(path)
        _spawn_detached(["explorer.exe", win_path])
        return f"explorer.exe {win_path}"
    webbrowser.open(path.as_uri())
    return f"webbrowser {path.as_uri()}"


def main() -> None:
    here = Path(__file__).resolve().parent
    args = parse_args()
    paths = args.paths or [here / "sphere.pt", here / "cube.pt"]
    shapes = {path.stem: load_shape(path) for path in paths}
    figure = build_figure(shapes)

    html_path = here / "viewer.html"
    figure.write_html(html_path, auto_open=False)
    opener = open_html(html_path)
    print(f"saved: {html_path}")
    print(f"opened: {opener}")


if __name__ == "__main__":
    main()
