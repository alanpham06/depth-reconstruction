"""Non-learned sparse-to-dense baselines, the bar the UNet has to clear.

  constant  every pixel gets the mean sparse depth (the network's `ref`)
  nearest   every pixel copies the depth of its nearest sparse point in the
            image plane (piecewise-constant Voronoi fill)

Both take the network's input and return depth maps shaped like its output,
so they are scored with exactly the same metrics.
"""

import torch


def constant_fill(inputs: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """inputs (B, 2, H, W), ref (B, 1, 1, 1) -> (B, 1, H, W)."""
    return ref.expand(-1, 1, *inputs.shape[-2:]).clone()


def nearest_fill(
    inputs: torch.Tensor, ref: torch.Tensor, chunk: int = 8192
) -> torch.Tensor:
    """Copy the nearest sparse pixel's depth into every pixel."""
    batch, _, height, width = inputs.shape
    v, u = torch.meshgrid(
        torch.arange(height, device=inputs.device),
        torch.arange(width, device=inputs.device),
        indexing="ij",
    )
    pixels = torch.stack([v, u], dim=-1).reshape(-1, 2).float()
    out = constant_fill(inputs, ref)
    for b in range(batch):
        mask = inputs[b, 1].reshape(-1) > 0
        if not mask.any():
            continue
        depth = (inputs[b, 0] + ref[b, 0]).reshape(-1)[mask]
        seeds = pixels[mask]
        # Direct distances, not cdist's matmul shortcut: the shortcut multiplies the
        # squared norms, which TF32 would round, and pixel distances are exact
        # without it on any device
        nearest = torch.cat(
            [
                torch.cdist(
                    block, seeds, compute_mode="donot_use_mm_for_euclid_dist"
                ).argmin(dim=1)
                for block in pixels.split(chunk)
            ]
        )
        out[b, 0] = depth[nearest].reshape(height, width)
    return out


BASELINES = {"constant": constant_fill, "nearest": nearest_fill}
