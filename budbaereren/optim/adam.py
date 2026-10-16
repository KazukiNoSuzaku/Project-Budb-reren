"""Adam (Kingma & Ba, 2015), mask-aware, with per-element step counts.

    m_t = β1·m + (1-β1)·g              first moment (EMA of gradients)
    v_t = β2·v + (1-β2)·g²             second moment (EMA of squared gradients)
    m̂ = m_t / (1-β1^t),  v̂ = v_t / (1-β2^t)      bias correction (m, v start at 0)
    w  -= lr · m̂ / (√v̂ + ε)

Dividing by √v̂ normalises each coordinate by its own RMS gradient, so the step
size is ~lr regardless of gradient scale. Weight decay, if used, is decoupled
(AdamW, Loshchilov & Hutter 2019).

Why the step count ``t`` is a per-element array, not a scalar: when a pruned
connection is revived its ``m`` and ``v`` are reset to 0. Bias correction
must then restart too. With a large *global* t, (1-β^t) ≈ 1 and the first
step is lr·(1-β1)/√(1-β2) ≈ 3.16·lr instead of lr. A per-element t restarted
at 0 makes a revived entry behave exactly like a freshly created parameter.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from budbaereren.nn.module import Parameter
from budbaereren.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.beta1, self.beta2 = betas
        self.eps, self.weight_decay = eps, weight_decay
        super().__init__(params, lr)

    def _init_state(self, p: Parameter) -> dict[str, np.ndarray]:
        return {
            "m": np.zeros_like(p.data),
            "v": np.zeros_like(p.data),
            "t": np.zeros(p.shape, dtype=np.int64),
        }

    def _bias_corrections(self, t: np.ndarray, dtype) -> tuple[np.ndarray, np.ndarray]:
        """Look up 1 - β^t for every entry from a cached table (β^t via pow is slow)."""
        needed = int(t.max()) + 1
        table = getattr(self, "_bc_table", None)
        if table is None or len(table[0]) < needed or table[0].dtype != dtype:
            size = max(needed, 2 * (0 if table is None else len(table[0])), 1024)
            steps = np.arange(size, dtype=np.float64)
            bc1 = (1.0 - self.beta1**steps).astype(dtype)
            bc2 = (1.0 - self.beta2**steps).astype(dtype)
            bc1[0] = bc2[0] = 1.0  # t = 0 only occurs for inactive entries; avoid 0/0
            self._bc_table = table = (bc1, bc2)
        return table[0][t], table[1][t]

    def _update(self, p, g, state, idx) -> None:
        m, v, t = state["m"], state["v"], state["t"]
        if idx is ...:
            active = None
            t += 1
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)
        else:
            # Arithmetic masking on full arrays instead of boolean indexing (which copies).
            # With a ∈ {0, 1}:  m + (1-β)(g - m)·a  equals  β·m + (1-β)·g  where a = 1 and
            # leaves m bit-for-bit unchanged where a = 0 (it adds an exact zero).
            active = idx.astype(p.data.dtype)
            t += idx
            m += (1.0 - self.beta1) * (g - m) * active
            v += (1.0 - self.beta2) * (g * g - v) * active

        bc1, bc2 = self._bias_corrections(t, p.data.dtype)
        update = (self.lr * (m / bc1)) / (np.sqrt(v / bc2) + self.eps)
        if self.weight_decay:
            update += self.lr * self.weight_decay * p.data
        if active is not None:
            update *= active
        p.data -= update
