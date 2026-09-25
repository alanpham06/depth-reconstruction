# Working in this repo

Sparse-to-dense depth completion on procedurally rendered ground truth: sparse
surface points in, a dense depth map out. `README.md` has the stages and the
commands; this file is how to move around.

## Layout

| | |
|---|---|
| `train.py` | the only entry point that trains |
| `arguments/`, `configs/` | the flag surface, and the yaml a `--config` name resolves to |
| `datasets/` | `sparse_depth.py` — capture files, the train/val split, augmentation, panel batches |
| `models/` | `unet.py` (the network, importable into a bare torch env), `losses.py`, `baselines.py`, `reconstruction_module.py` |
| `render/` | `capture.py`, and what it draws on: `geometry.py`, `raycast.py`, `sampling.py`, `point_cloud.py` |
| `eval/`, `export/` | `eval.py` and `export.py`, both CLIs |
| `plotting/`, `utils/` | the point-cloud figures, and the shared plumbing |
| `scripts/` | the cluster job |
| `tests/` | `python -m pytest tests` |
| `checkpoints/`, `runs/`, `output/`, `logs/`, `data/` | gitignored |

## Things that bite

**Read `val/mae` against `val/mae_nearest`, never on its own.** Copying each pixel's
nearest sparse point already fills a smooth shape well, so an absolute MAE says
little; the gap between the two is what the network is worth. `val/*_constant` is
what predicting the mean sparse depth everywhere scores.

**The val metrics are pooled over pixels, not averaged over batches.**
`SparseDepthModule` sums `depth_metrics` over the split and divides once, the way
`evaluate()` did on main, so `eval.py`'s "all" row reproduces the `val/mae` a
`best.ckpt` was chosen on -- for a run trained at 32-true, the precision eval
always scores at. `tests/test_training_run.py` pins that equality at 32-true.
Under the shipped bf16-mixed recipe the two differ by precision alone: measured
0.004993 against a training-logged 0.005201, and 0.004545 against 0.004603, on
the two equivalence checkpoints; scoring the first under bf16 instead of
float32 reads 0.005207.

**`--resume` needs the same `--epochs`.** The OneCycle schedule is built from the
run's length, and `train.check_resume` refuses anything else.

**Paths resolve from the repo root, not from `__file__`'s package.** `utils/paths.py`
goes up two levels and anchors `data/` there, so a capture or a training run reads
and writes the same `data/` regardless of the working directory. `runs/`,
`checkpoints/` and `output/` are relative to the directory a command runs in, not
to the repo root; the README's commands all assume that directory is the repo
root. Getting the repo-root resolution wrong raises nothing: the capture writes to
`render/data`, and nothing reads it. `tests/test_repo_layout.py` pins it.

**`models/__init__.py`, `models/unet.py`, `datasets/sparse_depth.py` and `utils/{paths,checkpoint}.py` import only
torch.** The export runs in the `executorch` conda env, which has neither Lightning
nor matplotlib. That is why the panel drawing lives in `utils/panel.py` and not in
the module.

**`nearest_fill` computes distances directly.** `train.py` enables TF32 matmuls, and
cdist's default matmul shortcut would feed the squared pixel norms (up to 130,050 at
256²) through a matmul that keeps 11 significant bits, which is enough to pick a
different nearest point. `tests/test_baselines.py` pins exactness on the GPU.

**Exports are GPU only.** An op the Vulkan partitioner leaves on the CPU fallback is
an error, and no `.pte` is written. Such an op fails at load time in an app built
with `EXECUTORCH_BUILD_PORTABLE_OPS=OFF`.

## Conventions

- Python style: `ruff check .` and `ruff format --check .`, using the 88-column
  settings in `ruff.toml`.
- One commit, one short sentence, no attribution lines.
- A claim about a number belongs next to the number that produced it. Keep the
  docstrings that carry measurements true, or delete them.

## Code style

Above every rule here: make the code simple and easy to understand. The plainest
construct that works wins.

### Docstrings and comments
- A module's docstring says what the file does and why, in full sentences, not a
  parameter list. An entry point names itself first: `"""train.py — <what it
  trains, in one line>."""`.
- A training entry point's docstring draws the data flow as a one-line arrow diagram
  and has a `Recipe` section with the exact command that runs it.
- A claim is a bold lead sentence, and the rest of the paragraph gives the
  measurement or mechanism behind it. Keep a docstring's numbers true or delete
  them: a stale measurement is worse than none.
- Inline comments explain why, not what. A comment that restates the code goes.
- A config value is commented with the measurement or reason that picked it, right
  there in the yaml.

### Naming
- A name carries its role, not just a noun: `grad_weight`, `checkpoint_dir`,
  `val_every`.
- Module-level constants are ALL_CAPS, with a comment saying where the value came
  from.
- A test name is a full sentence describing the behaviour it pins, not the method
  under test: `test_a_bool_cannot_stand_in_for_a_number`.

### Structure
- `train.py` is the only entry point that trains; everything it calls is a library
  or a CLI under a package. Flags live in `arguments/`; a config is a yaml file under
  `configs/` that `--config` resolves by name, and any flag on the command line beats
  the config.
- A config inherits with `extends: base`, and `configs/base.yaml` is the recipe every
  run shares unless it deliberately overrides a value.
- One file, one job: a losses module holds loss terms, a paths module holds where
  things live. There is no line cap, but no file mixes two jobs.
- Paths resolve from the repo root, not from `__file__`'s own package. Getting this
  wrong raises nothing — a directory just reads as empty — so a test pins it.
- Checkpoints, runs, outputs, logs and data are gitignored; nothing under them is
  committed.

### Errors and output
- Bad input raises at once and names the offending value:
  `raise ValueError(f"config {key} must be one of {allowed}, got {value!r}")`.
- A CLI-level problem is `raise SystemExit(...)` with a message that says what is
  wrong and what it would have cost: `--epochs must match the resumed run (100): the
  LR schedule depends on it`.
- Nothing fails silently. A wrong convention or path that "doesn't raise" is the
  failure to design against, because it reads as a worse result instead of an error.
- Status goes to stdout with plain `print(...)`, not `logging`; a line that must show
  before a slow step uses `flush=True`.

### Tests
- `python -m pytest tests` is the whole suite. A test module's docstring tells the
  story of the bug it pins, in prose, before any code.
- A test function rarely needs a docstring beyond its sentence-name; when it has
  one, it is a clause or two.
- Helpers are plain functions, often `_`-prefixed, rather than fixtures; `tmp_path`
  and `monkeypatch` cover the rest. Float comparisons state their tolerance
  (`pytest.approx`, `atol`/`rtol`), never a bare `==`.
- A test skips only when a local resource is genuinely absent; a suite that skips
  for any other reason has gone quiet, which is a bug.

### Formatting
- Ruff: 88 columns, 4-space indent, double quotes, lf line endings,
  `target-version = "py311"`, lint rules `["E4", "E7", "E9", "F", "I"]`.
  `.editorconfig` holds the same numbers for other files.
- A new module outside `tests/` starts with `from __future__ import annotations`.
- `pathlib.Path` throughout, never `os.path`.

### Commits
- One commit, one short lowercase sentence saying what the code now does — no ticket
  number, no prefix: `pin the cuda 12.8 build of torch`.
- No attribution lines of any kind: no `Co-Authored-By`, no "Generated with". The
  message is only the message.
