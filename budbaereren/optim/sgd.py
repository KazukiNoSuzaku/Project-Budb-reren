"""SGD with (heavy-ball) momentum, mask-aware."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from budbaereren.nn.module import Parameter
from budbaereren.optim.optimizer import Optimizer


class SGD(Optimizer):
    """``buf = μ·buf + g (+ λ·w);  w -= lr·buf`` on active entries only."""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ):
        self.momentum, self.weight_decay = momentum, weight_decay
        super().__init__(params, lr)

    def _init_state(self, p: Parameter) -> dict[str, np.ndarray]:
        return {"momentum": np.zeros_like(p.data)}

    def _update(self, p, g, state, idx) -> None:
        buf = state["momentum"]
        grad = g[idx]
        if self.weight_decay:
            grad = grad + self.weight_decay * p.data[idx]
        buf[idx] = self.momentum * buf[idx] + grad
        p.data[idx] -= self.lr * buf[idx]
