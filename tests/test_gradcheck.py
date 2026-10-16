"""Finite-difference gradient checks for every differentiable op in the engine.

Run just this suite with:  uv run pytest tests/test_gradcheck.py
"""

import numpy as np
import pytest

from budbaereren.engine import Tensor, ops
from budbaereren.engine.grad_check import check_gradients, numerical_grad

TOL = 1e-7  # observed worst case is ~1e-10
RNG = np.random.default_rng(42)


def randn(*shape):
    return RNG.standard_normal(shape)


def positive(*shape):
    return RNG.uniform(0.5, 2.0, shape)


def away_from_zero(*shape, margin=0.1):
    """For kinked ops (ReLU): keep inputs far from the kink relative to h = 1e-5."""
    x = RNG.uniform(margin, 2.0, shape)
    return x * RNG.choice([-1.0, 1.0], shape)


# (name, function of Tensors, list of input arrays)
CASES = [
    # element-wise binary, including broadcasting
    ("add", lambda a, b: a + b, [randn(4, 3), randn(4, 3)]),
    ("add_broadcast_bias", lambda a, b: a + b, [randn(4, 3), randn(3)]),
    ("sub_broadcast_col", lambda a, b: a - b, [randn(4, 3), randn(4, 1)]),
    ("mul", lambda a, b: a * b, [randn(4, 3), randn(4, 3)]),
    ("mul_broadcast_outer", lambda a, b: a * b, [randn(4, 1), randn(1, 3)]),
    ("div", lambda a, b: a / b, [randn(4, 3), positive(4, 3)]),
    ("div_broadcast", lambda a, b: a / b, [randn(4, 3), positive(3)]),
    ("neg", lambda a: -a, [randn(4, 3)]),
    # linear algebra
    ("matmul", lambda a, b: a @ b, [randn(5, 4), randn(4, 3)]),
    ("matmul_vector_like", lambda a, b: a @ b, [randn(1, 4), randn(4, 1)]),
    # reductions
    ("sum_all", lambda a: a.sum(), [randn(4, 3)]),
    ("sum_axis0", lambda a: a.sum(axis=0), [randn(4, 3)]),
    ("sum_axis1_keepdims", lambda a: a.sum(axis=1, keepdims=True), [randn(4, 3)]),
    ("sum_negative_axis_3d", lambda a: a.sum(axis=-1), [randn(2, 4, 3)]),
    ("sum_multi_axis", lambda a: a.sum(axis=(0, 2)), [randn(2, 4, 3)]),
    ("mean_all", lambda a: a.mean(), [randn(4, 3)]),
    ("mean_axis0", lambda a: a.mean(axis=0), [randn(4, 3)]),
    ("mean_axis1_keepdims", lambda a: a.mean(axis=1, keepdims=True), [randn(4, 3)]),
    # non-linearities
    ("relu", lambda a: a.relu(), [away_from_zero(4, 3)]),
    ("tanh", lambda a: a.tanh(), [randn(4, 3)]),
    ("sigmoid", lambda a: a.sigmoid(), [randn(4, 3) * 3]),
    ("gelu", lambda a: a.gelu(), [randn(4, 3) * 2]),
    ("exp", lambda a: a.exp(), [randn(4, 3)]),
    ("log", lambda a: a.log(), [positive(4, 3)]),
    ("pow_square", lambda a: a**2, [randn(4, 3)]),
    ("pow_fractional", lambda a: a**1.5, [positive(4, 3)]),
    # loss
    (
        "softmax_cross_entropy",
        lambda z: ops.softmax_cross_entropy(z, np.array([0, 2, 1, 2, 0])),
        [randn(5, 3)],
    ),
    # composite: one hidden layer MLP forward + loss, gradients w.r.t. every parameter
    (
        "mlp_composite",
        lambda x, w1, b1, w2, b2: ops.softmax_cross_entropy(
            ((x @ w1 + b1).tanh() @ w2) + b2, np.array([1, 0, 3, 2, 1, 0])
        ),
        [randn(6, 5), randn(5, 7) * 0.5, randn(7), randn(7, 4) * 0.5, randn(4)],
    ),
    (
        "masked_linear_composite",
        lambda x, w: ((x @ (w * Tensor((np.arange(12) % 3 != 0).reshape(4, 3)))).gelu()).mean(),
        [randn(5, 4), randn(4, 3)],
    ),
]


@pytest.mark.parametrize("name, fn, inputs", CASES, ids=[c[0] for c in CASES])
def test_gradient_matches_finite_differences(name, fn, inputs):
    errors = check_gradients(fn, inputs)
    assert max(errors) < TOL, f"{name}: relative errors {errors}"


def test_every_public_op_is_covered():
    """Guard against adding an op without adding a gradient check for it."""
    differentiable = {
        "add", "sub", "mul", "div", "neg", "matmul", "sum", "mean", "relu",
        "tanh", "sigmoid", "gelu", "exp", "log", "pow", "softmax_cross_entropy",
    }  # fmt: skip
    case_names = " ".join(c[0] for c in CASES)
    for op in differentiable:
        stem = {"softmax_cross_entropy": "softmax_cross_entropy", "pow": "pow_"}.get(op, op)
        assert stem in case_names, f"no gradient check for op '{op}'"


# ---------------------------------------------------------------------- softmax-CE specifics
def test_softmax_cross_entropy_value_matches_naive_formula():
    z = randn(6, 4)
    y = np.array([0, 1, 2, 3, 0, 1])
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    expected = -np.log(p[np.arange(6), y]).mean()
    assert ops.softmax_cross_entropy(Tensor(z), y).item() == pytest.approx(expected, rel=1e-12)


def test_softmax_cross_entropy_gradient_is_probs_minus_onehot_over_n():
    z = Tensor(randn(5, 3), requires_grad=True)
    y = np.array([2, 0, 1, 1, 2])
    ops.softmax_cross_entropy(z, y).backward()
    expected = ops.softmax(z.data)
    expected[np.arange(5), y] -= 1.0
    np.testing.assert_allclose(z.grad, expected / 5, rtol=1e-12)


def test_softmax_cross_entropy_is_stable_for_huge_logits():
    z = Tensor(np.array([[1000.0, 0.0, -1000.0], [-5e4, 5e4, 0.0]]), requires_grad=True)
    loss = ops.softmax_cross_entropy(z, np.array([0, 1]))
    loss.backward()
    assert np.isfinite(loss.item()) and loss.item() == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.isfinite(z.grad))


def test_softmax_cross_entropy_wrong_class_huge_logit_is_finite():
    z = Tensor(np.array([[1000.0, 0.0]]), requires_grad=True)
    loss = ops.softmax_cross_entropy(z, np.array([1]))
    loss.backward()
    assert loss.item() == pytest.approx(1000.0)
    np.testing.assert_allclose(z.grad, [[1.0, -1.0]])


def test_sigmoid_does_not_overflow():
    s = Tensor(np.array([-1e4, 0.0, 1e4])).sigmoid().data
    np.testing.assert_allclose(s, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------- the checker itself
def test_grad_checker_detects_a_wrong_gradient():
    """A gradient checker that cannot fail is worthless: plant a bug and catch it."""

    def buggy_square(a):
        return Tensor._from_op(a.data**2, (a,), lambda g: (g * a.data,), "buggy")  # should be 2a

    errors = check_gradients(lambda a: buggy_square(a).sum(), [positive(3)])
    assert errors[0] > 0.1


def test_numerical_grad_restores_input():
    x = Tensor(randn(3, 2), requires_grad=True)
    before = x.data.copy()
    numerical_grad(lambda: (x * x).sum(), x)
    np.testing.assert_array_equal(x.data, before)
