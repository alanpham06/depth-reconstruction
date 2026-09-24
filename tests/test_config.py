"""A config is a set of defaults, not an override.

A flag typed on the command line always beats the config, so a one-off run stays
a single flag on top of a named recipe. A config is checked exactly as the command
line would have been, since yaml reads 1e-3 as a string and true as a bool. And
running with no config at all is running the shared recipe.
"""

import sys

import pytest
import yaml

import train
from arguments.config import CONFIG_DIR, apply_config, load_config
from train import build_parser

REQUIRED = ["--train", "data/train"]
SHIPPED = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def _args(argv):
    return build_parser().parse_args([*REQUIRED, *argv])


def _write(tmp_path, name, body):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_there_are_configs_to_check():
    assert SHIPPED == ["base", "unet_b32"]


@pytest.mark.parametrize("name", SHIPPED)
def test_every_shipped_config_only_sets_real_options(name):
    """A typo in a config would otherwise sit there quietly doing nothing"""
    apply_config(_args([]), load_config(name), set(), build_parser())


def test_no_config_is_the_shared_recipe():
    """The parser's defaults are base.yaml's numbers, or a run without --config
    quietly trains something else."""
    defaults = vars(_args([]))
    for key, value in load_config("base").items():
        got = defaults[key]
        assert (list(got) if isinstance(got, tuple) else got) == value, key


def test_the_baseline_config_only_names_the_model():
    assert load_config("unet_b32") == {**load_config("base"), "base_channels": 32}


def test_a_flag_beats_the_config(tmp_path):
    config = load_config(_write(tmp_path, "wide", {"base_channels": 64}))
    args = _args(["--base-channels", "16"])
    apply_config(args, config, {"base_channels"}, build_parser())
    assert args.base_channels == 16


def test_the_config_fills_what_the_command_line_left_alone(tmp_path):
    config = load_config(_write(tmp_path, "wide", {"base_channels": 64}))
    args = _args([])
    assert apply_config(args, config, set(), build_parser()) == ["base_channels"]
    assert args.base_channels == 64


def test_a_range_reaches_its_flag_as_two_numbers(tmp_path):
    config = load_config(_write(tmp_path, "sparse", {"keep_range": [0.1, 0.5]}))
    args = _args([])
    apply_config(args, config, set(), build_parser())
    assert args.keep_range == [0.1, 0.5]


def test_scientific_notation_reaches_a_float_flag(tmp_path):
    config = load_config(_write(tmp_path, "rate", {"lr": "1e-3"}))
    args = _args([])
    apply_config(args, config, set(), build_parser())
    assert args.lr == pytest.approx(1e-3)


def test_a_bool_cannot_stand_in_for_a_number(tmp_path):
    config = load_config(_write(tmp_path, "oops", {"epochs": True}))
    with pytest.raises(ValueError, match="epochs"):
        apply_config(_args([]), config, set(), build_parser())


def test_a_config_cannot_set_an_option_that_does_not_exist(tmp_path):
    config = load_config(_write(tmp_path, "typo", {"base_chanels": 64}))
    with pytest.raises(ValueError, match="base_chanels"):
        apply_config(_args([]), config, set(), build_parser())


def test_a_choice_is_checked_the_way_the_flag_would_have_been(tmp_path):
    config = load_config(_write(tmp_path, "bad", {"precision": "fp8"}))
    with pytest.raises(ValueError, match="precision"):
        apply_config(_args([]), config, set(), build_parser())


def test_the_name_defaults_to_the_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED, "--config", "unet_b32"])
    assert train.parse_args().name == "unet_b32"
    monkeypatch.setattr(sys, "argv", ["train.py", *REQUIRED])
    assert train.parse_args().name == "run"


def test_mixed_precision_without_cuda_falls_back_to_full(capsys):
    assert train.resolve_precision("bf16-mixed", cuda=False) == "32-true"
    assert "32-true" in capsys.readouterr().out
    assert train.resolve_precision("bf16-mixed", cuda=True) == "bf16-mixed"
    assert train.resolve_precision("32-true", cuda=False) == "32-true"
