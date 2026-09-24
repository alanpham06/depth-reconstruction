"""Paths resolved from __file__ point at the repo root, not at the package.

generate_gt.py wrote to Path(__file__).parent / "data", which was the repo's data/
while it sat at the top level. Moved into render/, the same expression names
render/data -- a directory nothing reads, and nothing raises to say so.
"""

import sys
from pathlib import Path

import render.capture as capture
import render.point_cloud as point_cloud
import utils.paths as paths

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_data_resolves_to_the_repo_root():
    assert paths.REPO == REPO_ROOT
    assert paths.DATA == REPO_ROOT / "data"


def test_the_capture_writes_where_training_reads(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["capture"])
    assert capture.parse_args().out == REPO_ROOT / "data"


def test_the_point_clouds_land_in_data(tmp_path, monkeypatch):
    monkeypatch.setattr(point_cloud, "DATA", tmp_path)
    point_cloud.main()
    assert {p.name for p in tmp_path.iterdir()} == {"sphere.pt", "cube.pt"}


def test_the_config_dir_resolves_to_the_repo_root():
    import arguments.config as config

    assert config.CONFIG_DIR == REPO_ROOT / "configs"
    assert (config.CONFIG_DIR / "base.yaml").is_file()
