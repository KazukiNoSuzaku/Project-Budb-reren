"""Pruning: criteria, schedule shape, exact allocation, honest masking during training."""

import numpy as np
import pytest

from budbaereren.engine import Tensor, ops
from budbaereren.nn import MLP, Parameter
from budbaereren.optim import Adam
from budbaereren.prune import (
    CubicSchedule,
    OneShotSchedule,
    Pruner,
    allocate_masks,
    make_criterion,
    sparsity_of,
)


# ---------------------------------------------------------------------- schedule
def test_cubic_schedule_endpoints_and_shape():
    s = CubicSchedule(0.9, begin_step=100, end_step=1100, frequency=100)
    assert s.sparsity(0) == 0.0 and s.sparsity(100) == 0.0
    assert s.sparsity(1100) == pytest.approx(0.9) and s.sparsity(5000) == pytest.approx(0.9)
    values = [s.sparsity(t) for t in range(100, 1101, 100)]
    increments = np.diff(values)
    assert np.all(increments > 0), "monotone increasing"
    assert np.all(np.diff(increments) < 0), "concave: big steps early, small steps late"
    # Closed form at the midpoint: s_f (1 - (1/2)^3) = 0.9 * 7/8
    assert s.sparsity(600) == pytest.approx(0.9 * 7 / 8)


def test_cubic_schedule_snaps_end_to_a_prune_event():
    s = CubicSchedule(0.8, begin_step=430, end_step=5160, frequency=100)
    assert s.end_step == 5130 and s.is_prune_step(5130) and not s.is_prune_step(5160)
    assert s.sparsity(s.end_step) == pytest.approx(0.8)


def test_prune_steps_respect_frequency():
    s = CubicSchedule(0.5, begin_step=10, end_step=60, frequency=25)
    assert [t for t in range(100) if s.is_prune_step(t)] == [10, 35, 60]


def test_one_shot_schedule():
    s = OneShotSchedule(0.9, step=500)
    assert [t for t in range(1000) if s.is_prune_step(t)] == [500]
    assert s.sparsity(499) == 0.0 and s.sparsity(500) == 0.9


# ---------------------------------------------------------------------- allocation
def test_global_allocation_keeps_exactly_n_highest_scores():
    rng = np.random.default_rng(0)
    scores = [rng.random((20, 10)), rng.random((10, 5))]
    current = [np.ones((20, 10), bool), np.ones((10, 5), bool)]
    masks = allocate_masks(scores, current, n_keep=60, min_density=0.0)
    assert sum(m.sum() for m in masks) == 60
    kept = np.concatenate([s[m] for s, m in zip(scores, masks, strict=True)])
    dropped = np.concatenate([s[~m] for s, m in zip(scores, masks, strict=True)])
    assert kept.min() >= dropped.max()


def test_global_allocation_floor_prevents_layer_collapse():
    scores = [np.full((10, 10), 10.0), np.full((10, 10), 0.1)]  # layer 2 loses globally
    current = [np.ones((10, 10), bool)] * 2
    masks = allocate_masks(scores, current, n_keep=50, min_density=0.05)
    assert masks[1].sum() == 5 and masks[0].sum() == 45


def test_layerwise_allocation_gives_equal_density():
    rng = np.random.default_rng(0)
    scores = [rng.random((40, 10)), rng.random((10, 10))]
    current = [np.ones(s.shape, bool) for s in scores]
    masks = allocate_masks(scores, current, n_keep=100, scope="layerwise")
    assert [int(m.sum()) for m in masks] == [80, 20]


def test_allocation_never_revives_dead_weights():
    scores = [np.arange(10.0).reshape(2, 5)]
    current = [np.array([[True] * 5, [False] * 5])]  # the high scores are already dead
    masks = allocate_masks(scores, current, n_keep=3, min_density=0.0)
    assert not masks[0][1].any() and masks[0].sum() == 3


def test_allocation_rejects_impossible_budget():
    with pytest.raises(ValueError):
        allocate_masks([np.ones(4)], [np.array([True, True, False, False])], n_keep=3)


# ---------------------------------------------------------------------- criteria
def _param_with_grads(grads, w):
    p = Parameter(np.asarray(w, dtype=np.float64), prunable=True)
    return p, [np.asarray(g, dtype=np.float64) for g in grads]


def test_criteria_formulas_against_hand_computation():
    w = [2.0, -1.0, 0.5]
    grads = [[1.0, 1.0, -2.0], [3.0, -1.0, -2.0]]
    beta = 0.5
    g_ema = 0.5 * np.array(grads[0]) + 0.5 * np.array(grads[1])  # first obs initialises
    g2_ema = 0.5 * np.array(grads[0]) ** 2 + 0.5 * np.array(grads[1]) ** 2
    w = np.array(w)
    expected = {
        "magnitude": np.abs(w),
        "taylor1": np.abs(w * g_ema),
        "fisher": 0.5 * w**2 * g2_ema,
        "taylor2": np.abs(-w * g_ema + 0.5 * w**2 * g2_ema),
    }
    for name, exp in expected.items():
        crit = make_criterion(name, beta=beta)
        p, gs = _param_with_grads(grads, w)
        for g in gs:
            p.grad = g
            crit.observe([p])
        np.testing.assert_allclose(crit.scores(p), exp, err_msg=name)


def test_gradient_criteria_refuse_to_score_without_observations():
    p = Parameter(np.ones(3), prunable=True)
    with pytest.raises(RuntimeError):
        make_criterion("fisher").scores(p)


def test_fisher_prunes_weights_with_no_gradient_signal_before_large_ones():
    """A large weight that receives no gradient signal (e.g. from a dead input unit):
    magnitude ranks it highest, a loss-aware criterion knows it is useless."""
    p, gs = _param_with_grads([[0.0, 1.0, 1.0]] * 5, [5.0, 0.1, 0.2])
    fisher, mag = make_criterion("fisher"), make_criterion("magnitude")
    for g in gs:
        p.grad = g
        fisher.observe([p])
    assert np.argmin(fisher.scores(p)) == 0
    assert np.argmax(mag.scores(p)) == 0


# ---------------------------------------------------------------------- end to end
@pytest.mark.parametrize("criterion", ["magnitude", "taylor1", "fisher", "taylor2", "random"])
def test_pruner_hits_target_exactly_with_real_zeros(criterion):
    rng = np.random.default_rng(0)
    model = MLP([20, 32, 16, 4], rng, dtype=np.float64)
    opt = Adam(model.parameters(), lr=1e-2)
    x, y = rng.standard_normal((64, 20)), rng.integers(0, 4, 64)
    kwargs = {"seed": 0} if criterion == "random" else {}
    pruner = Pruner(
        model.prunable_parameters(),
        make_criterion(criterion, **kwargs),
        CubicSchedule(0.9, begin_step=10, end_step=110, frequency=10),
    )
    history = []
    for step in range(1, 151):
        opt.zero_grad()
        ops.softmax_cross_entropy(model(Tensor(x)), y).backward()
        opt.step()
        pruner(step)
        history.append(pruner.sparsity())
        weights = model.prunable_parameters()
        # Pruned entries are exact zeros in storage at every step, not just in the mask.
        assert all(np.all(p.data[~p.mask] == 0.0) for p in weights)

    n = sum(p.size for p in model.prunable_parameters())
    assert pruner.sparsity() == pytest.approx(1 - round(0.1 * n) / n)
    assert sparsity_of(model.prunable_parameters()) >= pruner.sparsity()
    assert np.all(np.diff(history) >= 0), "sparsity never decreases without regrowth"
    assert len(pruner.events) == 10  # 11 scheduled steps; the first has target 0 (no-op)
    for e in pruner.events:
        assert e.achieved == pytest.approx(e.target, abs=1.5 / n)


def test_pruned_network_still_learns():
    rng = np.random.default_rng(1)
    centers = rng.standard_normal((4, 20)) * 3
    y = rng.integers(0, 4, 256)
    x = centers[y] + rng.standard_normal((256, 20))
    model = MLP([20, 64, 4], rng, dtype=np.float64)
    opt = Adam(model.parameters(), lr=1e-2)
    pruner = Pruner(
        model.prunable_parameters(), make_criterion("fisher"), CubicSchedule(0.9, 20, 120, 10)
    )
    for step in range(1, 301):
        opt.zero_grad()
        ops.softmax_cross_entropy(model(Tensor(x)), y).backward()
        opt.step()
        pruner(step)
    acc = (model(Tensor(x)).data.argmax(1) == y).mean()
    assert pruner.sparsity() > 0.89 and acc > 0.95
