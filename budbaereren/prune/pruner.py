"""The pruner: turns criterion scores and a schedule into masks during training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from budbaereren.nn.module import Parameter
from budbaereren.prune.criteria import Criterion


def sparsity_of(params: list[Parameter]) -> float:
    """Fraction of prunable weights whose *stored value* is exactly zero."""
    total = sum(p.size for p in params)
    zeros = sum(int(np.count_nonzero(p.data == 0.0)) for p in params)
    return zeros / total


def mask_sparsity(params: list[Parameter]) -> float:
    total = sum(p.size for p in params)
    return 1.0 - sum(p.num_active() for p in params) / total


def allocate_masks(
    scores: list[np.ndarray],
    current: list[np.ndarray],
    n_keep: int,
    scope: str = "global",
    min_density: float = 0.01,
) -> list[np.ndarray]:
    """Choose exactly ``n_keep`` weights to keep, among those currently alive.

    * ``global``: one ranking across all layers. Each layer first keeps its top
      ``ceil(min_density * size)`` weights (a floor against layer collapse); the
      remaining budget goes to the highest scores anywhere.
    * ``layerwise``: every layer keeps the same fraction of its weights.

    Ties are broken deterministically (stable sort on index order).
    """
    sizes = [s.size for s in scores]
    alive = [c.ravel() for c in current]
    n_alive = sum(int(a.sum()) for a in alive)
    if not 0 <= n_keep <= n_alive:
        raise ValueError(f"cannot keep {n_keep} of {n_alive} alive weights")
    # Dead weights can never be chosen (masks are monotone while pruning).
    flat = [np.where(a, s.ravel(), -np.inf) for s, a in zip(scores, alive, strict=True)]

    if scope == "layerwise":
        total = sum(sizes)
        quotas = [int(round(n_keep * size / total)) for size in sizes]
        quotas[-1] += n_keep - sum(quotas)  # make the total exact
        quotas = [min(q, int(a.sum())) for q, a in zip(quotas, alive, strict=True)]
        keep = [_top_k(f, q) for f, q in zip(flat, quotas, strict=True)]
        shortfall = n_keep - sum(int(k.sum()) for k in keep)
        if shortfall:  # only possible if some layer had fewer alive than its quota
            keep = _fill_globally(flat, keep, shortfall)
    elif scope == "global":
        floors = [
            min(int(np.ceil(min_density * size)), int(a.sum()))
            for size, a in zip(sizes, alive, strict=True)
        ]
        if sum(floors) > n_keep:
            floors = [int(n_keep * f / sum(floors)) for f in floors]
        keep = [_top_k(f, k) for f, k in zip(flat, floors, strict=True)]
        keep = _fill_globally(flat, keep, n_keep - sum(floors))
    else:
        raise ValueError(f"unknown scope {scope!r}")

    return [k.reshape(c.shape) for k, c in zip(keep, current, strict=True)]


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    keep = np.zeros(scores.size, dtype=bool)
    if k > 0:
        keep[np.argsort(-scores, kind="stable")[:k]] = True
    return keep


def _fill_globally(flat: list[np.ndarray], keep: list[np.ndarray], budget: int) -> list[np.ndarray]:
    if budget <= 0:
        return keep
    candidates = np.concatenate([np.where(k, -np.inf, f) for f, k in zip(flat, keep, strict=True)])
    chosen = np.argsort(-candidates, kind="stable")[:budget]
    combined = np.concatenate(keep)
    combined[chosen] = True
    splits = np.cumsum([f.size for f in flat])[:-1]
    return np.split(combined, splits)


@dataclass
class PruneEvent:
    step: int
    target: float
    achieved: float
    layer_density: list[float] = field(default_factory=list)
    regrown: int = 0


class Pruner:
    """Step callback that prunes ``params`` according to ``schedule`` and ``criterion``.

    Called after every optimizer step: it feeds the criterion the fresh
    gradients and, on prune steps, installs new masks via ``Parameter.set_mask``
    (which writes exact zeros). The optimizer notices mask changes on its next
    step (see ``optim/optimizer.py``).
    """

    def __init__(
        self,
        params: list[Parameter],
        criterion: Criterion,
        schedule,
        scope: str = "global",
        min_density: float = 0.01,
    ):
        if not params or any(not p.prunable for p in params):
            raise ValueError("Pruner needs a non-empty list of prunable parameters")
        self.params = params
        self.criterion = criterion
        self.schedule = schedule
        self.scope = scope
        self.min_density = min_density
        self.n_weights = sum(p.size for p in params)
        self.events: list[PruneEvent] = []

    def __call__(self, step: int) -> None:
        self.criterion.observe(self.params)
        if self.schedule.is_prune_step(step):
            self.prune_to(self.schedule.sparsity(step), step)

    def prune_to(self, target: float, step: int = -1) -> PruneEvent | None:
        n_keep = int(round((1.0 - target) * self.n_weights))
        n_alive = sum(p.num_active() for p in self.params)
        if n_keep >= n_alive:
            return None  # schedule has not yet asked for more than we already pruned
        scores = [self.criterion.scores(p) for p in self.params]
        masks = allocate_masks(
            scores, [p.mask for p in self.params], n_keep, self.scope, self.min_density
        )
        for p, m in zip(self.params, masks, strict=True):
            p.set_mask(m)
        achieved = sparsity_of(self.params)
        # Exactness check on the stored weights, not just the masks. A weight that
        # is alive but happens to be exactly 0.0 would make storage sparsity exceed
        # the mask sparsity; that is benign, so only the other direction is an error.
        if achieved < 1.0 - n_keep / self.n_weights - 1e-12:
            raise AssertionError(f"pruned weights are not exact zeros (sparsity {achieved})")
        event = PruneEvent(step, target, mask_sparsity(self.params), self.layer_density())
        self.events.append(event)
        return event

    def layer_density(self) -> list[float]:
        return [p.num_active() / p.size for p in self.params]

    def sparsity(self) -> float:
        return mask_sparsity(self.params)
