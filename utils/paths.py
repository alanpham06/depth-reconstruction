"""Where things live, so every stage agrees without being told twice.

    data/<split>/<shape>_gt.pt  what render.capture writes: one file per
                                primitive, every view stacked. Gitignored; the
                                capture regenerates it in about two minutes
    data/cube.pt, sphere.pt     render.point_cloud's point clouds, which
                                plotting/ draws
    output/<tool>/              anything derived: evaluations, exports, figures

Resolved from the repository root rather than from this file's package, so a
module that moves between packages cannot quietly start pointing at a directory
that does not exist.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
# Relative: a tool's output lands under the directory it was run from
OUTPUT = Path("output")
