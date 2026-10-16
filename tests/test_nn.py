"""Layers, modules and initialisation."""

import numpy as np
import pytest

from budbaereren.engine import Tensor, ops
from budbaereren.engine.grad_check import numerical_grad, relative_error
from budbaereren.nn import MLP, Linear
from budbaereren.nn.init import he_normal, xavier_uniform


def test_linear_forward_matches_numpy():
    rng = np.random.default_rng(0)
    layer = Linear(4, 3, rng, dtype=np.float64)
    x = rng.standard_normal((5, 4))
    out = layer(Tensor(x))
    np.testing.assert_allclose(out.data, x @ layer.weight.data + layer.bias.data)


def test_mlp_parameter_discovery_and_shapes():
    model = MLP([784, 300, 100, 10], np.random.default_rng(0))
    names = [n for n, _ in model.named_parameters()]
    assert names == [
        "net.layers.0.weight", "net.layers.0.bias",
        "net.layers.2.weight", "net.layers.2.bias",
        "net.layers.4.weight", "net.layers.4.bias",
    ]  # fmt: skip
    assert model.num_parameters() == 784 * 300 + 300 + 300 * 100 + 100 + 100 * 10 + 10
    assert [p.shape for p in model.prunable_parameters()] == [(784, 300), (300, 100), (100, 10)]
    assert all(p.mask is None for _, p in model.named_parameters() if p.name == "bias")


def test_mlp_gradients_match_finite_differences():
    """End-to-end check of the composed layers, not just the individual ops."""
    rng = np.random.default_rng(1)
    model = MLP([5, 7, 4], rng, activation="tanh", dtype=np.float64)
    x, y = rng.standard_normal((6, 5)), np.array([0, 1, 2, 3, 0, 1])

    def loss():
        return ops.softmax_cross_entropy(model(Tensor(x)), y)

    loss().backward()
    for p in model.parameters():
        assert relative_error(p.grad, numerical_grad(loss, p)) < 1e-7


@pytest.mark.parametrize("fan_in, fan_out", [(784, 300), (300, 100)])
def test_he_init_variance(fan_in, fan_out):
    w = he_normal(fan_in, fan_out, np.random.default_rng(0), dtype=np.float64)
    assert w.var() == pytest.approx(2.0 / fan_in, rel=0.05)


def test_xavier_init_variance():
    w = xavier_uniform(100, 10, np.random.default_rng(0), dtype=np.float64)
    assert w.var() == pytest.approx(2.0 / 110, rel=0.1)


def test_he_init_preserves_activation_scale_through_relu_layers():
    """The reason for He init: second moment stays ~constant with depth under ReLU."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2048, 256))
    second_moments = []
    for _ in range(6):
        x = np.maximum(x @ he_normal(256, 256, rng, dtype=np.float64), 0.0)
        second_moments.append(np.mean(x**2))
    assert max(second_moments) / min(second_moments) < 1.5


def test_state_dict_roundtrip_includes_masks():
    rng = np.random.default_rng(0)
    a, b = MLP([4, 3, 2], rng), MLP([4, 3, 2], rng)
    w = a.prunable_parameters()[0]
    mask = np.ones(w.shape, dtype=bool)
    mask[0, 0] = False
    w.set_mask(mask)
    b.load_state_dict(a.state_dict())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        np.testing.assert_array_equal(pa.data, pb.data)
    assert not b.prunable_parameters()[0].mask[0, 0]
