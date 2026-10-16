"""Regrowth: drop-and-grow mechanics, decay schedule, and stability during training."""

import numpy as np
import pytest

from budbaereren.engine import Tensor, ops
from budbaereren.nn import MLP
from budbaereren.optim import Adam
from budbaereren.prune import (
    CubicSchedule,
    Pruner,
    RegrowthSchedule,
    drop_and_grow,
    jaccard,
    make_criterion,
)


def test_drop_and_grow_swaps_lowest_active_for_highest_dead():
    keep = [np.array([5.0, 1.0, 3.0, 0.0, 0.0, 0.0])]
    grow = [np.array([0.0, 9.0, 0.0, 2.0, 7.0, 1.0])]
    mask = [np.array([True, True, True, False, False, False])]
    new, n = drop_and_grow(keep, grow, mask, n_swap=1)
    assert n == 1
    # Drops index 1 (lowest keep score among active); grows index 4 (highest grow score
    # among weights that were dead *before* this event). Index 1 is not re-grown even
    # though its grow score (9) is the highest.
    np.testing.assert_array_equal(new[0], [True, False, True, False, True, False])


def test_drop_and_grow_preserves_count_across_layers():
    rng = np.random.default_rng(0)
    masks = [rng.random((30, 20)) < 0.2, rng.random((20, 5)) < 0.3]
    keep = [rng.random(m.shape) for m in masks]
    grow = [rng.random(m.shape) for m in masks]
    new, n = drop_and_grow(keep, grow, masks, n_swap=25)
    assert n == 25
    assert sum(m.sum() for m in new) == sum(m.sum() for m in masks)
    assert jaccard(masks, new) < 1.0


def test_drop_and_grow_respects_eligibility():
    keep = [np.array([5.0, 1.0, 0.0, 0.0])]
    grow = [np.array([0.0, 0.0, 9.0, 1.0])]
    mask = [np.array([True, True, False, False])]
    eligible = [np.array([False, False, False, True])]  # index 2 was pruned this very step
    new, _ = drop_and_grow(keep, grow, mask, 1, eligible)
    np.testing.assert_array_equal(new[0], [True, False, False, True])


def test_regrowth_schedule_decays_to_zero():
    s = RegrowthSchedule(0.3, begin_step=100, end_step=1100, frequency=100)
    assert s.fraction(100) == pytest.approx(0.3)
    assert s.fraction(600) == pytest.approx(0.15)
    assert s.fraction(1100) == 0.0 and not s.is_event(1100)
    fractions = [s.fraction(t) for t in range(100, 1100, 100)]
    assert np.all(np.diff(fractions) < 0)


def test_jaccard():
    a = [np.array([True, True, False, False])]
    b = [np.array([True, False, True, False])]
    assert jaccard(a, a) == 1.0 and jaccard(a, b) == pytest.approx(1 / 3)


def test_criteria_see_dense_gradient_of_pruned_weights_but_score_them_zero():
    rng = np.random.default_rng(0)
    model = MLP([6, 8, 3], rng, dtype=np.float64)
    w = model.prunable_parameters()[0]
    mask = np.ones(w.shape, bool)
    mask[0, :] = False
    w.set_mask(mask)
    crit = make_criterion("fisher")
    x, y = rng.standard_normal((16, 6)), rng.integers(0, 3, 16)
    ops.softmax_cross_entropy(model(Tensor(x)), y).backward()
    crit.observe([w])
    assert np.all(w.grad[0] == 0.0)  # parameter gradient: exactly zero
    assert np.all(crit._stat(w, "g2")[0] > 0)  # dense statistics: informative
    assert np.all(crit.scores(w)[0] == 0.0)  # but w = 0, so the pruning score is 0


def test_training_with_regrowth_is_stable_and_honest():
    rng = np.random.default_rng(2)
    centers = rng.standard_normal((4, 20)) * 3
    y = rng.integers(0, 4, 256)
    x = centers[y] + rng.standard_normal((256, 20))
    model = MLP([20, 64, 4], rng, dtype=np.float64)
    opt = Adam(model.parameters(), lr=1e-2)
    weights = model.prunable_parameters()
    pruner = Pruner(
        weights,
        make_criterion("fisher"),
        CubicSchedule(0.9, 20, 120, 10),
        regrowth=RegrowthSchedule(0.3, begin_step=20, end_step=200, frequency=10),
    )
    revived_total, sparsities = 0, []
    for step in range(1, 301):
        opt.zero_grad()
        ops.softmax_cross_entropy(model(Tensor(x)), y).backward()
        opt.step()
        revived_total += opt.revived_last_step
        pruner(step)
        sparsities.append(pruner.sparsity())
        assert all(np.all(p.data[~p.mask] == 0.0) for p in weights)

    assert revived_total > 0, "regrowth should have revived some connections"
    # Every connection the pruner grew was seen as a revival by the optimizer, so every
    # one of them started with fresh Adam state. (This failed before grow candidates
    # were restricted to weights dead *before* the step: a weight pruned and regrown in
    # the same step kept its stale state.)
    assert sum(e.regrown for e in pruner.regrowth_events) == revived_total
    # Sparsity is held exactly at the target once reached, despite the churn.
    target = sparsities[120]
    assert all(s == pytest.approx(target) for s in sparsities[120:])
    # Churn decays: the last events change the mask far less than the busiest one.
    # (The very first event swaps nothing: no weight is dead yet, so none can grow.)
    churn = [1 - e.jaccard for e in pruner.regrowth_events]
    assert max(churn) > 0 and churn[-1] < 0.25 * max(churn)
    acc = (model(Tensor(x)).data.argmax(1) == y).mean()
    assert acc > 0.95
