"""Masked-weight correctness: forward, gradient, and optimizer state.

What we guarantee (see docs/DESIGN.md, "The gradient of a masked weight"):

* A masked weight contributes exactly nothing to the forward pass.
* Its *parameter* gradient ∂L/∂W is exactly 0 (chain rule through W ⊙ M), while
  the *dense* gradient ∂L/∂(W⊙M) stays available for regrowth decisions.
* The optimizer neither moves it nor touches its state while it is pruned.
* On revival its optimizer state restarts from scratch, including Adam's
  bias-correction step count, so its first update is exactly that of a fresh
  parameter.

Each guarantee is contrasted with a plausible naive implementation that
violates it, so the tests show *why* the guarantee matters.
"""

import numpy as np
import pytest

from budbaereren.engine import Tensor, ops
from budbaereren.engine.grad_check import numerical_grad
from budbaereren.nn import MLP, Linear, Parameter
from budbaereren.optim import SGD, Adam

LR = 1e-2


# ---------------------------------------------------------------------- fixtures
def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    model = MLP([6, 8, 3], rng, dtype=np.float64)
    x = rng.standard_normal((32, 6))
    y = rng.integers(0, 3, 32)
    return model, x, y


def train_steps(model, opt, x, y, n, on_step=None):
    for _ in range(n):
        opt.zero_grad()
        ops.softmax_cross_entropy(model(Tensor(x)), y).backward()
        opt.step()
        if on_step is not None:
            on_step()


def prune_entries(param: Parameter, entries):
    mask = param.mask.copy()
    for e in entries:
        mask[e] = False
    param.set_mask(mask)
    return ~mask


# ---------------------------------------------------------------------- naive baselines
class MaskUnawareAdam(Adam):
    """The common quick implementation: rely on the masked (zero) gradient alone."""

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is not None:
                self._update(p, p.grad, self.state[i], ...)  # whole tensor, ignores mask


class StaleStateAdam(Adam):
    """Respects the mask while pruned, but never resets state on revival."""

    def _reset_state(self, state, where):
        pass


class GlobalStepAdam(Adam):
    """Resets m, v on revival but uses one global step count for bias correction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_t = 0

    def step(self):
        self.global_t += 1
        super().step()

    def _update(self, p, g, state, idx):
        state["t"][idx] = self.global_t  # every entry shares the global count
        state["t"][idx] -= 1  # (parent _update increments it back)
        super()._update(p, g, state, idx)


# ---------------------------------------------------------------------- forward
def test_masked_weight_contributes_exactly_nothing_to_forward():
    rng = np.random.default_rng(0)
    layer = Linear(5, 4, rng, dtype=np.float64)
    pruned = prune_entries(layer.weight, [(0, 0), (2, 3), (4, 1)])
    assert np.all(layer.weight.data[pruned] == 0.0)  # real zero in storage

    # Even if something writes garbage into pruned storage, the forward is unaffected.
    layer.weight.data[pruned] = 1e6
    x = rng.standard_normal((7, 5))
    expected = x @ (layer.weight.data * layer.weight.mask) + layer.bias.data
    np.testing.assert_array_equal(layer(Tensor(x)).data, expected)


# ---------------------------------------------------------------------- gradient
def test_parameter_gradient_is_exactly_zero_and_dense_gradient_is_retained():
    rng = np.random.default_rng(0)
    layer = Linear(5, 4, rng, dtype=np.float64)
    pruned = prune_entries(layer.weight, [(0, 0), (2, 3), (4, 1)])
    x, y = rng.standard_normal((7, 5)), rng.integers(0, 4, 7)

    ops.softmax_cross_entropy(layer(Tensor(x)), y).backward()

    # Parameter gradient: exactly zero (not 1e-12) at pruned entries.
    assert np.all(layer.weight.grad[pruned] == 0.0)
    # Dense gradient ∂L/∂(W⊙M): non-zero at pruned entries ...
    assert np.all(np.abs(layer.dense_grad[pruned]) > 0)
    # ... and equal, at every entry, to the gradient of an unmasked layer whose
    # weights are W⊙M, i.e. "what if this connection existed (at value 0)?".
    w_eff = layer.weight.data * layer.weight.mask
    dense = Linear(5, 4, rng, dtype=np.float64)
    dense.weight.data[...] = w_eff
    dense.bias.data[...] = layer.bias.data
    loss = lambda: ops.softmax_cross_entropy(dense(Tensor(x)), y)  # noqa: E731
    np.testing.assert_allclose(layer.dense_grad, numerical_grad(loss, dense.weight), rtol=1e-6)
    # Active entries: parameter gradient == dense gradient.
    np.testing.assert_array_equal(layer.weight.grad[~pruned], layer.dense_grad[~pruned])


# ---------------------------------------------------------------------- optimizer, while pruned
@pytest.mark.parametrize("make_opt", [lambda p: Adam(p, lr=LR), lambda p: SGD(p, lr=LR)])
def test_pruned_weights_stay_exactly_zero_and_state_is_frozen(make_opt):
    model, x, y = make_problem()
    opt = make_opt(model.parameters())
    train_steps(model, opt, x, y, 20)

    w = model.prunable_parameters()[0]
    pruned = prune_entries(w, [(0, 0), (1, 3), (5, 7), (3, 2)])
    w_index = opt.params.index(w)
    frozen = {k: v[pruned].copy() for k, v in opt.state[w_index].items()}

    def check():
        assert np.all(w.data[pruned] == 0.0), "pruned weight is not an exact zero"
        for k, v in opt.state[w_index].items():
            np.testing.assert_array_equal(v[pruned], frozen[k], err_msg=f"state '{k}' changed")

    train_steps(model, opt, x, y, 50, on_step=check)
    # Active weights did train.
    assert not np.allclose(opt.state[w_index][next(iter(frozen))][~pruned], 0.0)


def test_naive_mask_unaware_adam_lets_pruned_weights_drift():
    """Masking only the gradient is NOT enough: Adam's momentum keeps moving the weight."""
    model, x, y = make_problem()
    opt = MaskUnawareAdam(model.parameters(), lr=LR)
    train_steps(model, opt, x, y, 20)
    w = model.prunable_parameters()[0]
    pruned = prune_entries(w, [(0, 0), (1, 3), (5, 7), (3, 2)])
    assert np.all(w.data[pruned] == 0.0)

    grads_at_pruned = []
    train_steps(model, opt, x, y, 50, on_step=lambda: grads_at_pruned.append(w.grad[pruned]))
    # The gradient was exactly zero all along, yet the "pruned" weights moved.
    assert np.all(np.concatenate(grads_at_pruned) == 0.0)
    assert np.max(np.abs(w.data[pruned])) > 1e-3


# ---------------------------------------------------------------------- optimizer, on revival
def isolated_param_run(opt_cls, steps_before=5000, steps_pruned=50, g_before=1.0, g_after=-1.0):
    """One prunable parameter, a controlled gradient sequence, and a revival.

    Before pruning the gradient is ``g_before``; after revival it is ``g_after``.
    Returns the weight change produced by the first post-revival step.
    """
    p = Parameter(np.zeros((2, 2)), prunable=True)
    opt = opt_cls([p], lr=LR)
    for _ in range(steps_before):
        p.grad = np.full(p.shape, g_before)
        opt.step()
    mask = np.ones((2, 2), dtype=bool)
    mask[0, 0] = False
    p.set_mask(mask)
    for _ in range(steps_pruned):
        p.grad = np.full(p.shape, g_before) * p.mask  # masked gradient is exactly 0
        opt.step()
    assert p.data[0, 0] == 0.0
    p.set_mask(np.ones((2, 2), dtype=bool))  # revive
    p.grad = np.full(p.shape, g_after)
    before = p.data[0, 0]
    opt.step()
    return p.data[0, 0] - before, opt


def test_revived_weight_first_update_equals_fresh_parameter():
    delta, opt = isolated_param_run(Adam)
    # A fresh Adam parameter's first step is exactly lr·sign(-g) (for |g| ≫ eps).
    assert delta == pytest.approx(+LR, rel=1e-6)
    assert opt.revived_last_step == 1
    assert opt.state[0]["t"][0, 0] == 1  # its own bias-correction clock restarted


def test_global_step_count_oversizes_first_update_after_revival():
    """Resetting m, v but keeping a global t: bias correction is ~1, step ≈ 3.16·lr."""
    delta, _ = isolated_param_run(GlobalStepAdam)
    expected = LR * (1 - 0.9) / np.sqrt(1 - 0.999)  # = 3.162·lr for t ≫ 1/(1-β2)
    assert delta == pytest.approx(expected, rel=0.02)
    assert abs(delta) > 3 * LR


def test_stale_state_pushes_revived_weight_the_wrong_way():
    """Without a reset, momentum from before pruning dominates the new gradient.

    The new gradient is -1 (so the weight should *increase*), but m still holds
    ≈ +1 from before pruning, and the first step goes the wrong way.
    """
    delta, _ = isolated_param_run(StaleStateAdam)
    assert delta < 0


@pytest.mark.parametrize("opt_cls", [Adam, SGD])
def test_revival_resets_all_state_buffers(opt_cls):
    p = Parameter(np.zeros(3), prunable=True)
    opt = opt_cls([p], lr=LR)
    for _ in range(10):
        p.grad = np.ones(3)
        opt.step()
    p.set_mask(np.array([True, False, True]))
    p.grad = np.ones(3) * p.mask
    opt.step()
    p.set_mask(np.ones(3, dtype=bool))
    p.grad = np.zeros(3)  # zero gradient: state of the revived entry must stay at reset values
    opt.step()
    for name, buf in opt.state[0].items():
        if name != "t":
            assert buf[1] == 0.0, f"{opt_cls.__name__} state '{name}' not reset on revival"
