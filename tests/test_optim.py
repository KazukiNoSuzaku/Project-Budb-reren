"""Optimizer correctness on dense parameters (masking is covered in test_masking.py)."""

import numpy as np
import pytest

from budbaereren.nn import Parameter
from budbaereren.optim import SGD, Adam


def reference_adam(grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, w0=0.0):
    """Line-by-line transcription of Algorithm 1 in Kingma & Ba (2015)."""
    w, m, v = w0, 0.0, 0.0
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w


def test_adam_matches_paper_algorithm():
    grads = np.random.default_rng(0).standard_normal(200)
    for prunable in (False, True):  # both code paths: mask=None and all-True mask
        p = Parameter(np.zeros(1), prunable=prunable)
        opt = Adam([p], lr=1e-2)
        for g in grads:
            p.grad = np.array([g])
            opt.step()
        assert p.data[0] == pytest.approx(reference_adam(grads, lr=1e-2), rel=1e-10)


def test_adam_first_step_is_lr_times_sign():
    p = Parameter(np.zeros(3))
    opt = Adam([p], lr=0.1)
    p.grad = np.array([5.0, -0.01, 1e3])
    opt.step()
    np.testing.assert_allclose(p.data, [-0.1, 0.1, -0.1], rtol=1e-6)


def test_sgd_momentum_matches_recurrence():
    p = Parameter(np.zeros(1))
    opt = SGD([p], lr=0.1, momentum=0.9)
    w, buf = 0.0, 0.0
    for g in [1.0, 2.0, -1.0, 0.5]:
        p.grad = np.array([g])
        opt.step()
        buf = 0.9 * buf + g
        w -= 0.1 * buf
        assert p.data[0] == pytest.approx(w)


@pytest.mark.parametrize("opt_cls, lr", [(Adam, 0.05), (SGD, 0.05)])
def test_optimizers_minimise_a_quadratic(opt_cls, lr):
    target = np.array([1.0, -2.0, 3.0])
    p = Parameter(np.zeros(3))
    opt = opt_cls([p], lr=lr)
    for _ in range(2000):
        p.grad = 2 * (p.data - target)
        opt.step()
    np.testing.assert_allclose(p.data, target, atol=1e-3)


def test_decoupled_weight_decay_skips_pruned_entries():
    p = Parameter(np.ones(4), prunable=True)
    p.set_mask(np.array([True, True, False, True]))
    opt = Adam([p], lr=0.1, weight_decay=0.5)
    p.grad = np.zeros(4)
    opt.step()
    assert p.data[2] == 0.0
    np.testing.assert_allclose(p.data[[0, 1, 3]], 1 - 0.1 * 0.5)


def test_parameters_without_grad_are_skipped():
    a, b = Parameter(np.ones(2)), Parameter(np.ones(2))
    opt = Adam([a, b], lr=0.1)
    a.grad = np.ones(2)
    opt.step()
    np.testing.assert_array_equal(b.data, [1.0, 1.0])
    assert not np.allclose(a.data, 1.0)
