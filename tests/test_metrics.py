"""Validation metrics are pooled over a split's pixels, not averaged per batch.

evaluate() on main summed errors and pixel counts over the whole split and then
divided once. Lightning's default epoch mean averages per-batch means instead,
which weights a batch of small silhouettes the same as a batch of large ones.
"""

import pytest
import torch

from utils.metrics import METRICS, PooledMetrics, depth_metrics, summarize_metrics


def _batch(error: float, valid_pixels: int):
    gt = torch.full((1, 1, 4, 4), 2.0)
    valid = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    valid.view(-1)[:valid_pixels] = True
    return gt + error, gt, valid, torch.zeros_like(valid)


def test_a_split_is_pooled_over_its_pixels_not_its_batches():
    pooled = PooledMetrics()
    pooled.add("model", *_batch(error=1.0, valid_pixels=2))
    pooled.add("model", *_batch(error=0.1, valid_pixels=14))
    mae = pooled.summarize()["model"]["mae"]
    assert mae == pytest.approx((2 * 1.0 + 14 * 0.1) / 16)
    assert mae != pytest.approx((1.0 + 0.1) / 2)


def test_each_predictor_is_pooled_on_its_own():
    pooled = PooledMetrics()
    pooled.add("model", *_batch(error=0.5, valid_pixels=4))
    pooled.add("nearest", *_batch(error=2.0, valid_pixels=4))
    summary = pooled.summarize()
    assert summary["model"]["mae"] == pytest.approx(0.5)
    assert summary["nearest"]["mae"] == pytest.approx(2.0)


def test_reduce_sees_every_total_before_the_division():
    """What DDP needs: every total summed across processes, then one division."""
    pooled = PooledMetrics()
    pooled.add("model", *_batch(error=1.0, valid_pixels=4))
    seen = []

    def double(total):
        seen.append(total)
        return total * 2

    summary = pooled.summarize(reduce=double)
    assert len(seen) == len(depth_metrics(*_batch(1.0, 4)))
    assert summary["model"]["mae"] == pytest.approx(1.0)  # both sides doubled


def test_the_metric_names_are_the_ones_summarize_returns():
    assert tuple(summarize_metrics(depth_metrics(*_batch(0.5, 4)))) == METRICS
