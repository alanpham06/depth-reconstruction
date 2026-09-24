"""nearest_fill runs wherever the batch is, and stays exact there.

It built its pixel grid on the CPU, which was fine while baselines were scored from
a CPU DataLoader. Validation now scores them on the batch Lightning has moved to
the GPU, where a CPU grid indexed with a CUDA mask raises. And train.py turns on
TF32 matmuls, which cdist's default matmul path would feed the squared pixel norms
through -- up to 130,050 at 256x256, far past the 11 significant bits TF32 keeps --
so an inexact distance could pick a different nearest point.
"""

import pytest
import torch

from datasets.sparse_depth import assemble_input
from models.baselines import constant_fill, nearest_fill


def _sparse_batch(size: int = 32, keep: float = 0.05, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    depth = 2.0 + torch.rand(2, size, size, generator=generator)
    mask = torch.rand(2, size, size, generator=generator) < keep
    return assemble_input(depth * mask, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_nearest_fill_is_exact_on_the_gpu_even_with_tf32():
    inputs, ref = _sparse_batch(size=256, keep=0.01)
    expected = nearest_fill(inputs, ref)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        got = nearest_fill(inputs.cuda(), ref.cuda())
    finally:
        torch.set_float32_matmul_precision(previous)
    assert got.device.type == "cuda"
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=0)


def test_a_view_with_no_sparse_points_falls_back_to_the_constant_fill():
    inputs, ref = _sparse_batch(keep=0.0)
    torch.testing.assert_close(nearest_fill(inputs, ref), constant_fill(inputs, ref))
