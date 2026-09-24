"""Sparse-to-dense training samples built from the data/<name>_gt.pt files.

Every view of every primitive becomes one sample:

  input   (2, H, W)  channel 0: sparse depth minus the sample's reference depth,
                     0 where there is no point; channel 1: binary valid-point mask
  target  (1, H, W)  dense ground-truth z-depth
  valid   (1, H, W)  pixels that hit the surface, the only ones the loss sees
  ref     (1, 1, 1)  reference depth (mean of the sparse points) that was
                     subtracted from the input; the network predicts depth
                     relative to it, so add it back to get metric depth

Centering on the sparse points' mean makes the input independent of how far
the camera is from the object: the network only has to learn shape, not
absolute distance.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


def assemble_input(
    sparse_depth: torch.Tensor, sparse_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack sparse depth and its mask into the network input.

    sparse_depth, sparse_mask: (..., H, W). Returns (input (..., 2, H, W),
    ref (..., 1, 1, 1)) where ref is the per-map mean of the valid depths.
    """
    mask = sparse_mask.to(sparse_depth.dtype)
    count = mask.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
    ref = (sparse_depth * mask).sum(dim=(-2, -1), keepdim=True) / count
    centered = (sparse_depth - ref) * mask
    return torch.stack([centered, mask], dim=-3), ref.unsqueeze(-3)


def load_views(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Concatenate the per-view tensors the network needs from several files."""
    keys = ("depth", "mask", "sparse_depth", "sparse_mask")
    parts = {key: [] for key in keys}
    for path in paths:
        data = torch.load(path, map_location="cpu")
        for key in keys:
            parts[key].append(data[key])
    return {key: torch.cat(value) for key, value in parts.items()}


class SparseDepthDataset(Dataset):
    """All views from a set of *_gt.pt files, with optional augmentation.

    Augmentation (training only) is applied to the stored sparse map rather
    than re-rendering, so it is free:

      keep_range   each sample keeps a uniformly drawn fraction of its sparse
                   points, so the network sees many densities from one render
      flip         random horizontal / vertical flips (a flipped depth map is
                   still a valid depth map of a mirrored scene)
      noise_std    Gaussian noise on the kept sparse depths, in scene units
    """

    def __init__(
        self,
        paths: list[Path],
        views: torch.Tensor | None = None,
        augment: bool = False,
        keep_range: tuple[float, float] = (0.3, 1.0),
        noise_std: float = 0.0,
    ):
        data = load_views(paths)
        if views is not None:
            data = {key: value[views] for key, value in data.items()}
        self.depth = data["depth"].float()
        self.mask = data["mask"].bool()
        self.sparse_depth = data["sparse_depth"].float()
        self.sparse_mask = data["sparse_mask"].bool()
        self.augment = augment
        self.keep_range = keep_range
        self.noise_std = noise_std

    def __len__(self) -> int:
        return self.depth.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        depth = self.depth[index]
        valid = self.mask[index]
        sparse_depth = self.sparse_depth[index]
        sparse_mask = self.sparse_mask[index]

        if self.augment:
            low, high = self.keep_range
            keep = low + (high - low) * torch.rand(())
            sparse_mask = sparse_mask & (torch.rand(sparse_mask.shape) < keep)
            if self.noise_std > 0:
                sparse_depth = sparse_depth + self.noise_std * torch.randn_like(
                    sparse_depth
                )
            sparse_depth = sparse_depth * sparse_mask
            for dim in (-1, -2):
                if torch.rand(()) < 0.5:
                    depth, valid, sparse_depth, sparse_mask = (
                        x.flip(dim) for x in (depth, valid, sparse_depth, sparse_mask)
                    )

        inputs, ref = assemble_input(sparse_depth, sparse_mask)
        return {
            "input": inputs,
            "target": depth.unsqueeze(0),
            "valid": valid.unsqueeze(0),
            "ref": ref,
        }
