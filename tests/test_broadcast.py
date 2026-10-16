"""Broadcasting must backpropagate correctly: the adjoint of a broadcast is a sum."""

import numpy as np
import pytest

from budbaereren.engine import Tensor, unbroadcast

RNG = np.random.default_rng(0)


@pytest.mark.parametrize(
    "grad_shape, target, expected_axes",
    [
        ((4, 3), (4, 3), None),  # no broadcasting
        ((4, 3), (3,), (0,)),  # new leading axis
        ((4, 3), (1, 3), (0,)),  # size-1 axis
        ((4, 3), (4, 1), (1,)),
        ((4, 3), (), (0, 1)),  # scalar
        ((2, 4, 3), (4, 1), (0, 2)),  # both kinds at once
    ],
)
def test_unbroadcast_sums_over_broadcast_axes(grad_shape, target, expected_axes):
    g = RNG.standard_normal(grad_shape)
    out = unbroadcast(g, target)
    assert out.shape == target
    expected = g if expected_axes is None else g.sum(axis=expected_axes).reshape(target)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [((5, 3), (3,)), ((5, 3), (1, 3)), ((5, 1), (1, 3)), ((5, 3), ()), ((3,), (5, 3))],
)
def test_binary_op_broadcast_gradients(op, shape_a, shape_b):
    """Compare each broadcast gradient against the same op on explicitly tiled inputs."""
    a_np = RNG.uniform(0.5, 2.0, shape_a)
    b_np = RNG.uniform(0.5, 2.0, shape_b)
    out_shape = np.broadcast_shapes(shape_a, shape_b)
    upstream = RNG.standard_normal(out_shape)

    a, b = Tensor(a_np, requires_grad=True), Tensor(b_np, requires_grad=True)
    fn = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
    }[op]
    fn(a, b).backward(upstream)

    # Reference: tile both operands to the full shape (no broadcasting), backprop,
    # then scatter-add each tiled gradient back to the element it was copied from.
    at = Tensor(np.broadcast_to(a_np, out_shape).copy(), requires_grad=True)
    bt = Tensor(np.broadcast_to(b_np, out_shape).copy(), requires_grad=True)
    fn(at, bt).backward(upstream)

    np.testing.assert_allclose(a.grad, _scatter_back(at.grad, shape_a), rtol=1e-12)
    np.testing.assert_allclose(b.grad, _scatter_back(bt.grad, shape_b), rtol=1e-12)


def _scatter_back(tiled_grad, shape):
    """Independent of unbroadcast: map each output position to its source element."""
    ref = np.zeros(shape)
    offset = tiled_grad.ndim - len(shape)
    for out_idx in np.ndindex(tiled_grad.shape):
        src = tuple(0 if shape[d] == 1 else out_idx[d + offset] for d in range(len(shape)))
        ref[src] += tiled_grad[out_idx]
    return ref


def test_bias_add_example_from_brief():
    """(N, D) + (D,): the bias gradient is the column sum of the upstream gradient."""
    x = Tensor(RNG.standard_normal((8, 4)), requires_grad=True)
    bias = Tensor(RNG.standard_normal(4), requires_grad=True)
    upstream = RNG.standard_normal((8, 4))
    (x + bias).backward(upstream)
    np.testing.assert_allclose(bias.grad, upstream.sum(axis=0))
    np.testing.assert_allclose(x.grad, upstream)
