"""Composable run configs, so an experiment is a file rather than a command line"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _resolve(path: Path, relative_to: Path | None = None) -> Path:
    """An inherited path means a sibling of the config that named it, then the shared directory

    The naming config wins over the working directory, so a stray file next to wherever
    the command ran cannot quietly replace a parent
    """
    candidate = Path(path)
    if not candidate.suffix:
        candidate = candidate.with_suffix(".yaml")
    if relative_to is not None:
        sibling = relative_to / candidate
        if sibling.exists():
            return sibling.resolve()
    elif candidate.exists():
        return candidate.resolve()
    shared = CONFIG_DIR / candidate
    if shared.exists():
        return shared.resolve()
    raise FileNotFoundError(f"config {path} not found")


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads a config and every config it extends, child values winning over parent"""
    resolved = _resolve(Path(path))
    seen: list[Path] = []
    chain: list[dict[str, Any]] = []

    current: Path | None = resolved
    while current is not None:
        if current in seen:
            names = " -> ".join(item.name for item in [*seen, current])
            raise ValueError(f"config extends in a cycle: {names}")
        seen.append(current)
        loaded = yaml.safe_load(current.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config {current.name} must be a mapping")
        parent = loaded.pop("extends", None)
        chain.append(loaded)
        if parent is None:
            current = None
        elif isinstance(parent, str):
            current = _resolve(Path(parent), current.parent)
        else:
            raise ValueError(f"config {current.name} extends a {type(parent).__name__}")

    # Parents first, so a child overrides what it inherits
    merged: dict[str, Any] = {}
    for entry in reversed(chain):
        merged.update(entry)
    return merged


def given_options(build_parser: Callable[..., argparse.ArgumentParser]) -> set[str]:
    """The options actually supplied on the command line, by dest.

    Parsing a second time with every default suppressed leaves only what was typed, so
    abbreviations, short flags and a dest that differs from the option all resolve the
    way argparse itself resolved them
    """
    sparse = build_parser()
    # argument_default only reaches arguments declared without one, so suppress each action
    for action in sparse._actions:
        action.default = argparse.SUPPRESS
    return set(vars(sparse.parse_args()))


def apply_config(
    args: Any,
    config: dict[str, Any],
    given: set[str],
    parser: argparse.ArgumentParser | None = None,
) -> list[str]:
    """Fills argparse values the command line did not set, and reports what it filled.

    An explicit flag always wins, so a config is a set of defaults rather than an
    override, and a one-off run stays a single flag on top of a named experiment.
    """
    actions = {action.dest: action for action in parser._actions} if parser else {}
    applied = []
    for key, value in sorted(config.items()):
        if not hasattr(args, key):
            raise ValueError(f"config sets unknown option {key}")
        if key in given:
            continue
        setattr(args, key, _coerce(args, key, value, actions.get(key)))
        applied.append(key)
    return applied


def _takes_no_value(action: argparse.Action) -> bool:
    """Store-true, store-false and --flag/--no-flag all set a bool without reading a value"""
    return isinstance(action, argparse.BooleanOptionalAction) or action.nargs == 0


def _takes_many_values(action: argparse.Action) -> bool:
    """nargs of +, * or a count above one collects a list rather than one value"""
    if isinstance(action.nargs, int):
        return action.nargs > 1
    return action.nargs in ("+", "*")


def _single_value(action: argparse.Action) -> argparse.Action:
    """The same action seen as holding one value, for coercing an entry of its list"""
    entry = copy.copy(action)
    entry.nargs = None
    return entry


def _coerce(
    args: Any, key: str, value: Any, action: argparse.Action | None = None
) -> Any:
    """Matches the type the flag would have produced, since yaml reads 1e-3 as a string

    Running the flag's own type and choices over the text form means a config is checked
    exactly as the command line would have been
    """
    if action is not None and _takes_no_value(action):
        # A bare flag has no value to parse, so only yaml's own true/false can mean one
        if not isinstance(value, bool):
            raise ValueError(f"config {key} must be true or false, got {value!r}")
        return value
    if action is not None and _takes_many_values(action):
        # An nargs flag holds a list, and coercing the list itself would hand the
        # flag's type the string "['a', 'b']", so each entry is coerced on its own
        if not isinstance(value, list):
            value = [value]
        return [_coerce(args, key, entry, _single_value(action)) for entry in value]
    if action is not None and action.type is not None:
        # Skipping the coercion for a bool let yaml's true reach a numeric flag as True,
        # which arithmetic silently treats as 1: steps: true trained for one step
        if isinstance(value, bool):
            kind = getattr(action.type, "__name__", "value")
            raise ValueError(f"config {key} takes a {kind}, not {str(value).lower()}")
        try:
            value = action.type(str(value))
        except (TypeError, ValueError) as error:
            kind = getattr(action.type, "__name__", "value")
            raise ValueError(f"config {key} is not a valid {kind}: {error}") from error
    if (
        action is not None
        and action.choices is not None
        and value not in action.choices
    ):
        allowed = ", ".join(str(choice) for choice in action.choices)
        raise ValueError(f"config {key} must be one of {allowed}, got {value!r}")
    if action is not None and action.type is not None:
        return value
    current = getattr(args, key)
    if isinstance(current, float) and isinstance(value, str):
        return float(value)
    if (
        isinstance(current, int)
        and not isinstance(current, bool)
        and isinstance(value, str)
    ):
        return int(value)
    return value
