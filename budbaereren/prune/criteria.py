"""Importance criteria: how a connection earns the right to survive.

Removing weight w_i is the perturbation δw = -w_i e_i. A second-order Taylor
expansion of the loss with a diagonal Hessian (Optimal Brain Damage) gives

    ΔL_i ≈ -g_i w_i + ½ H_ii w_i²

Each criterion below estimates (a part of) this quantity; higher score means
removing the weight is predicted to hurt more, so it is kept. See DESIGN.md §2
for the derivation and why each approximation is reasonable.

Criteria observe gradients every training step (``observe``) and are asked
for scores only at prune events (``scores``). Gradient statistics are EMAs,
because a single minibatch gradient is dominated by sampling noise.
"""

from __future__ import annotations

import numpy as np

from budbaereren.nn.module import Parameter


class Criterion:
    name = "base"

    def __init__(self, beta: float = 0.99):
        self.beta = beta
        self._stats: dict[int, dict[str, np.ndarray]] = {}

    def observe(self, params: list[Parameter]) -> None:
        """Update running gradient statistics from the current ``.grad``s."""

    def scores(self, p: Parameter) -> np.ndarray:
        raise NotImplementedError

    def _ema(self, p: Parameter, key: str, value: np.ndarray) -> None:
        stats = self._stats.setdefault(id(p), {})
        if key not in stats:
            stats[key] = value.astype(np.float64)  # first observation initialises the EMA
        else:
            stats[key] *= self.beta
            stats[key] += (1.0 - self.beta) * value

    def _stat(self, p: Parameter, key: str) -> np.ndarray:
        try:
            return self._stats[id(p)][key]
        except KeyError:
            raise RuntimeError(f"{self.name}: no gradients observed before scoring") from None


class Magnitude(Criterion):
    """|w|: the classic baseline (Han et al., 2015).

    In terms of the Taylor expansion it assumes g = 0 and the same curvature
    H_ii for every weight, so that ΔL ∝ w². Neither assumption holds during
    training, which is what the other criteria exploit.
    """

    name = "magnitude"

    def scores(self, p: Parameter) -> np.ndarray:
        return np.abs(p.data).astype(np.float64)


class TaylorFirstOrder(Criterion):
    """|w · E[g]|: magnitude of the first-order Taylor term.

    E[g] is estimated by an EMA of minibatch gradients. Valid while training is
    under way (E[g] ≠ 0). At a converged minimum E[g] → 0 and this term
    vanishes, which is exactly why OBD needed the second-order term.
    """

    name = "taylor1"

    def observe(self, params):
        for p in params:
            if p.grad is not None:
                self._ema(p, "g", p.grad)

    def scores(self, p):
        return np.abs(p.data * self._stat(p, "g"))


class Fisher(Criterion):
    """½ w² · E[g²]: the second-order (OBD) term, Hessian diagonal ≈ E[g²].

    For a log-likelihood loss, the Fisher information equals the expected
    Hessian (Gauss-Newton), and its diagonal is the expected squared gradient.
    We estimate it with an EMA of squared minibatch gradients, the same quantity
    Adam tracks as ``v``. Unlike the first-order term it does not vanish at a
    minimum, and unlike magnitude it scales each weight by how sensitive the
    loss is to it.
    """

    name = "fisher"

    def observe(self, params):
        for p in params:
            if p.grad is not None:
                self._ema(p, "g2", p.grad * p.grad)

    def scores(self, p):
        w = p.data.astype(np.float64)
        return 0.5 * w * w * self._stat(p, "g2")


class TaylorSecondOrder(Criterion):
    """|−w·E[g] + ½ w²·E[g²]|: both terms of the expansion."""

    name = "taylor2"

    def observe(self, params):
        for p in params:
            if p.grad is not None:
                self._ema(p, "g", p.grad)
                self._ema(p, "g2", p.grad * p.grad)

    def scores(self, p):
        w = p.data.astype(np.float64)
        return np.abs(-w * self._stat(p, "g") + 0.5 * w * w * self._stat(p, "g2"))


class Random(Criterion):
    """Uniformly random scores: the floor any criterion must beat (sanity check)."""

    name = "random"

    def __init__(self, beta: float = 0.99, seed: int = 0):
        super().__init__(beta)
        self.rng = np.random.default_rng(seed)

    def scores(self, p):
        return self.rng.random(p.shape)


CRITERIA: dict[str, type[Criterion]] = {
    c.name: c for c in (Magnitude, TaylorFirstOrder, Fisher, TaylorSecondOrder, Random)
}


def make_criterion(name: str, **kwargs) -> Criterion:
    try:
        return CRITERIA[name](**kwargs)
    except KeyError:
        raise ValueError(f"unknown criterion {name!r}; choose from {sorted(CRITERIA)}") from None
