"""Lightweight 2D UNet for sparse-to-dense depth completion.

Four downsampling stages with channel widths base, 2*base, 4*base, 8*base and
a 16*base bottleneck; at the default base=32 that is about 7.8M parameters.
Inputs whose sides are not a multiple of 16 are padded and the output is
cropped back, so any image size works.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(2 * out_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([self.up(x), skip], dim=1))


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base: int = 32,
        depth: int = 4,
    ):
        super().__init__()
        widths = [base * 2**level for level in range(depth + 1)]
        self.stem = DoubleConv(in_channels, widths[0])
        self.down = nn.ModuleList(
            DoubleConv(widths[i], widths[i + 1]) for i in range(depth)
        )
        self.up = nn.ModuleList(
            Up(widths[i + 1], widths[i]) for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(widths[0], out_channels, 1)
        self.multiple = 2**depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h, pad_w = -height % self.multiple, -width % self.multiple
        x = F.pad(x, (0, pad_w, 0, pad_h))

        skips = [self.stem(x)]
        for block in self.down:
            skips.append(block(F.max_pool2d(skips[-1], 2)))
        x = skips.pop()
        for block in self.up:
            x = block(x, skips.pop())
        # Keep the output layer in fp32 under autocast: bfloat16 would round the
        # predicted depth to ~0.4%, coarser than the errors we want to resolve.
        with torch.autocast(x.device.type, enabled=False):
            return self.head(x.float())[..., :height, :width]
