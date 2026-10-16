"""RigL-style regrowth: drop the least important, grow the most promising.

At each regrowth event (every ``frequency`` steps until ``end_step``) a
fraction ``α_t`` of the currently active weights is dropped (lowest criterion
score) and the *same number* of currently pruned connections is grown back,
so sparsity is unchanged. Grown connections are chosen by the EMA of their
squared **dense** gradient E[(∂L/∂W_eff)²]: the Fisher information of a
connection that does not exist yet, i.e. how strongly the loss "wants" it.

Grown weights start at exactly 0.0; the optimizer detects the 0→1 mask
transition and starts them with fresh Adam state (see optim/optimizer.py).

Stability: α_t decays with a cosine to 0 at ``end_step`` (Evci et al., 2020).
Without decay the mask keeps churning: every event removes weights that have
had only ``frequency`` steps to grow, and training never settles. Weights
dropped in an event are not eligible to be regrown in the same event, and
weights grown in an event are protected from the immediately following drop
by having had ``frequency`` steps to grow first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegrowthSchedule:
    initial_fraction: float = 0.3
    begin_step: int = 0
    end_step: int = 0
    frequency: int = 100

    def fraction(self, step: int) -> float:
        if step < self.begin_step or step >= self.end_step:
            return 0.0
        progress = (step - self.begin_step) / (self.end_step - self.begin_step)
        return self.initial_fraction * 0.5 * (1.0 + np.cos(np.pi * progress))

    def is_event(self, step: int) -> bool:
        return (
            self.begin_step <= step < self.end_step
            and (step - self.begin_step) % self.frequency == 0
        )


def drop_and_grow(
    keep_scores: list[np.ndarray],
    grow_scores: list[np.ndarray],
    masks: list[np.ndarray],
    n_swap: int,
    eligible: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Drop the ``n_swap`` lowest-scoring active weights and grow the ``n_swap``
    highest-scoring inactive ones (global ranking). Returns new masks, n grown.

    ``eligible`` restricts which inactive weights may grow. The pruner passes
    "dead before this training step", so a weight pruned at this very step
    cannot be grown straight back: the optimizer would see its mask go
    True → True, never detect a revival, and apply stale Adam state to a weight
    that was just reset to 0.0.
    """
    flat_masks = np.concatenate([m.ravel() for m in masks])
    can_grow = ~flat_masks
    if eligible is not None:
        can_grow &= np.concatenate([e.ravel() for e in eligible])
    n_active = int(flat_masks.sum())
    n_swap = min(n_swap, n_active, int(can_grow.sum()))
    if n_swap <= 0:
        return [m.copy() for m in masks], 0

    keep = np.concatenate([s.ravel() for s in keep_scores])
    grow = np.concatenate([s.ravel() for s in grow_scores])
    new = flat_masks.copy()

    active_idx = np.flatnonzero(flat_masks)
    drop = active_idx[np.argsort(keep[active_idx], kind="stable")[:n_swap]]
    grow_pool = np.flatnonzero(can_grow)  # excludes the weights dropped just now
    grow_idx = grow_pool[np.argsort(-grow[grow_pool], kind="stable")[:n_swap]]
    new[drop] = False
    new[grow_idx] = True

    splits = np.cumsum([m.size for m in masks])[:-1]
    return [p.reshape(m.shape) for p, m in zip(np.split(new, splits), masks, strict=True)], n_swap


def jaccard(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """|A ∩ B| / |A ∪ B| of two sets of active connections (1.0 = identical masks)."""
    inter = sum(int(np.logical_and(x, y).sum()) for x, y in zip(a, b, strict=True))
    union = sum(int(np.logical_or(x, y).sum()) for x, y in zip(a, b, strict=True))
    return inter / union if union else 1.0
