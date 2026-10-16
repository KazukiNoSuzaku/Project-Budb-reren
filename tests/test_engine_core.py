"""Core autodiff behaviour: hand-derived gradients, accumulation, graph traversal."""

import numpy as np
import pytest

from budbaereren.engine import Tensor, no_grad


def leaf(x):
    return Tensor(np.asarray(x, dtype=np.float64), requires_grad=True)


def test_elementwise_gradients_match_hand_derivation():
    a, b = leaf([2.0, -3.0]), leaf([4.0, 5.0])
    out = (a * b + a / b - b).data.sum()  # forward sanity
    assert out == pytest.approx((2 * 4 + 2 / 4 - 4) + (-3 * 5 + -3 / 5 - 5))

    loss = a * b + a / b - b
    loss.backward(np.ones(2))
    # d/da = b + 1/b ; d/db = a - a/b^2 - 1
    np.testing.assert_allclose(a.grad, [4 + 1 / 4, 5 + 1 / 5])
    np.testing.assert_allclose(b.grad, [2 - 2 / 16 - 1, -3 + 3 / 25 - 1])


def test_reflected_ops_and_negation():
    x = leaf([1.0, 2.0])
    y = 3.0 - x * 2.0 + 1.0 / x - (-x)
    y.backward(np.ones(2))
    # dy/dx = -2 - 1/x^2 + 1
    np.testing.assert_allclose(x.grad, [-2 - 1 + 1, -2 - 0.25 + 1])


def test_reused_node_accumulates_gradient():
    # y = x*x uses x twice: dy/dx = 2x. A '=' instead of '+=' would give x.
    x = leaf([3.0])
    (x * x).backward(np.ones(1))
    np.testing.assert_allclose(x.grad, [6.0])


def test_diamond_graph():
    #      x
    #     / \
    #    a   b      a = 2x, b = x^2, out = a*b = 2x^3  -> d/dx = 6x^2
    #     \ /
    #     out
    x = leaf([2.0])
    a = x * 2.0
    b = x * x
    (a * b).backward(np.ones(1))
    np.testing.assert_allclose(x.grad, [6 * 4.0])


def test_leaf_gradients_accumulate_across_backward_calls_until_zeroed():
    x = leaf([1.0])
    (x * 3.0).backward(np.ones(1))
    (x * 3.0).backward(np.ones(1))
    np.testing.assert_allclose(x.grad, [6.0])
    x.zero_grad()
    (x * 3.0).backward(np.ones(1))
    np.testing.assert_allclose(x.grad, [3.0])


def test_deep_graph_does_not_hit_recursion_limit():
    x = leaf([1.0])
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward(np.ones(1))
    np.testing.assert_allclose(x.grad, [1.0])


def test_constants_receive_no_gradient():
    x = leaf([1.0, 2.0])
    c = Tensor([5.0, 6.0])
    (x * c).backward(np.ones(2))
    assert c.grad is None
    np.testing.assert_allclose(x.grad, [5.0, 6.0])


def test_no_grad_builds_no_graph():
    x = leaf([1.0])
    with no_grad():
        y = x * 2.0
    assert not y.requires_grad and y.is_leaf


def test_retain_grad_exposes_intermediate_gradient():
    x = leaf([1.0, 2.0])
    h = (x * 3.0).retain_grad()
    (h * h).backward(np.ones(2))
    np.testing.assert_allclose(h.grad, 2 * h.data)
    np.testing.assert_allclose(x.grad, 2 * h.data * 3.0)


def test_backward_requires_explicit_grad_for_non_scalar():
    with pytest.raises(RuntimeError):
        (leaf([1.0, 2.0]) * 2.0).backward()


def test_python_scalars_preserve_float32():
    x = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    y = 2.0 * x + 1.0
    assert y.dtype == np.float32
    y.backward(np.ones(3, dtype=np.float32))
    assert x.grad.dtype == np.float32


def test_integer_inputs_are_promoted_to_float():
    assert Tensor([1, 2, 3]).dtype == np.float64
