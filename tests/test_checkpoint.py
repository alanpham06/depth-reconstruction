"""A checkpoint gives back its network at the width it was trained at, without Lightning."""

import torch

from models import count_parameters, make_model, predict
from utils.checkpoint import load_network


def test_the_network_comes_back_at_its_width_with_its_weights(tmp_path):
    torch.manual_seed(0)
    trained = make_model(base_channels=8)
    path = tmp_path / "run_version_0_best.ckpt"
    torch.save(
        {
            "state_dict": {f"model.{k}": v for k, v in trained.state_dict().items()},
            "hyper_parameters": {"base_channels": 8},
        },
        path,
    )
    network = load_network(path)
    assert count_parameters(network) == count_parameters(trained)
    assert not network.training
    inputs, ref = torch.randn(1, 2, 32, 32), torch.full((1, 1, 1, 1), 4.5)
    trained.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            predict(network, inputs, ref), predict(trained, inputs, ref), rtol=0, atol=0
        )


def test_predict_adds_the_reference_depth_back():
    network = make_model(base_channels=8).eval()
    inputs, ref = torch.randn(2, 2, 32, 32), torch.full((2, 1, 1, 1), 3.0)
    with torch.no_grad():
        torch.testing.assert_close(
            predict(network, inputs, ref), ref + network(inputs), rtol=0, atol=0
        )
