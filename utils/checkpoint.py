"""How a later script reads a finished run's network back.

eval.py and export.py rebuild the network from a checkpoint's weights and the
width it recorded, and never import the training wrapper, so they run where
Lightning is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models import make_model


def load_payload(path: str | Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a checkpoint")
    return raw


def load_network(path: str | Path, device: torch.device | None = None):
    """Just the network, at the width the run recorded, in eval mode.

    SparseDepthModule keeps the network as self.model, so its weights are the
    checkpoint's model.* entries.
    """
    payload = load_payload(path)
    weights = {
        key.removeprefix("model."): value
        for key, value in payload["state_dict"].items()
        if key.startswith("model.")
    }
    if not weights:
        raise ValueError(f"{path} holds no model.* weights")
    width = payload.get("hyper_parameters", {}).get("base_channels", 32)
    network = make_model(base_channels=int(width))
    network.load_state_dict(weights)
    return network.to(device or "cpu").eval()
