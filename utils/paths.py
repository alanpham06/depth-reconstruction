"""Where things live, so every stage agrees without being told twice.

    data/<split>/<shape>_gt.pt  what render.capture writes: one file per
                                primitive, every view stacked. Gitignored; the
                                capture regenerates it in about two minutes
    data/cube.pt, sphere.pt     render.point_cloud's point clouds, which
                                plotting/ draws
    runs/<run>/version_N/       one training run, versioned by Lightning: re-running
                                a name gets version_1 rather than overwriting
                                version_0. Events, samples/, hparams.yaml,
                                summary.json and config.json -- telemetry only
    checkpoints/                the models, beside runs/ rather than inside it, as
                                <run>_version_N_<best|last>.ckpt, so clearing runs/
                                to tidy TensorBoard never deletes one
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

CHECKPOINTS = "checkpoints"
SUMMARY = "summary.json"
RESOLVED_CONFIG = "config.json"
SAMPLES = "samples"

# What render.capture names a primitive's file, and what --train/--val look for
CAPTURE_SUFFIX = "_gt.pt"
CAPTURE_GLOB = f"*{CAPTURE_SUFFIX}"


def checkpoint_dir(run_dir: str | Path) -> Path:
    """<repo>/checkpoints, beside runs/ rather than inside it.

    runs/ is telemetry -- events, sample grids, the resolved config -- and deleting
    it to clear TensorBoard should not destroy the models. It did once: 1.5 GB of
    trained checkpoints went with a single `rm -rf runs/*`.

    Resolved by walking up out of runs/ rather than from the working directory, so
    a test handed a tmp_path writes into its own sandbox instead of the repo.
    """
    root = _runs_root(run_dir)
    if root is not None:
        return root.parent / CHECKPOINTS
    return Path(run_dir).resolve() / CHECKPOINTS


def run_stem(run_dir: str | Path) -> str:
    """<run>_version_N, taken from the run directory itself.

    The whole path below runs/, not just the last component: --output takes a
    directory that runs live in, so runs/ablation_a/myrun and runs/ablation_b/myrun
    are different runs. Keyed on the last component alone they produced one
    filename and the second silently overwrote the first.

    Unchanged for the ordinary runs/<name>/version_N shape, so checkpoints written
    before this still resolve.
    """
    run_dir = Path(run_dir).resolve()
    root = _runs_root(run_dir)
    if root is None or run_dir == root:
        return run_dir.name
    return "_".join(run_dir.relative_to(root).parts)


def run_name(checkpoint) -> str:
    """<run>_version_N, read off checkpoints/<run>_version_N_<best|last>.ckpt.

    A name in any other shape -- Lightning's own last.ckpt, a best-v1.ckpt --
    comes back as its stem, unchanged.
    """
    stem = Path(checkpoint).stem
    for suffix in ("_best", "_last"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def default_output(tool: str, checkpoint, source: str | None = None) -> Path:
    """output/<tool>/<run>[_<source>], the run's name taken from the checkpoint.

    The source is in the name because one run is scored against several splits,
    and the run name alone would put every one of them in the same directory.
    """
    name = run_name(checkpoint)
    if source is not None:
        name = f"{name}_{source}"
    return OUTPUT / tool / name


def source_label(paths) -> str:
    """What a set of captures is called in an output name: the directory they are in.

    data/val and data/val/cube_gt.pt are both `val`, so scoring one split twice,
    however it was named, writes one directory rather than two.
    """
    folders = {(p if p.is_dir() else p.parent).name for p in map(Path, paths)}
    return "_".join(sorted(folders))


def _runs_root(run_dir: Path) -> Path | None:
    """The OUTERMOST `runs` ancestor of a run directory, or None if there is none.

    Outermost, not nearest: with runs/a/runs/b the nearest match put the models in
    runs/a/checkpoints -- still inside the directory this scheme exists to make
    safe to delete.
    """
    parts = Path(run_dir).resolve().parts
    for i, name in enumerate(parts):
        if name == "runs":
            return Path(*parts[: i + 1])
    return None
