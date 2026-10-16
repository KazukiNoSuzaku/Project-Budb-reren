"""Mask-aware optimizer base class.

The masking contract, shared by every optimizer in this package:

1. Entries with ``mask == False`` are not updated at all: no parameter step,
   no optimizer-state update (no momentum decay), no weight decay. After every
   step the optimizer re-asserts ``param[~mask] = 0.0`` exactly.
2. State for a pruned entry is frozen and never read while it is pruned.
3. When an entry is revived (mask goes False -> True), its state is reset as
   if the parameter had just been created. Revival is *detected here* by
   diffing the mask against the one seen on the previous step, so no caller can
   forget to notify the optimizer.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from budbaereren.nn.module import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.params: list[Parameter] = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        self.lr = lr
        self.state: list[dict[str, np.ndarray]] = [self._init_state(p) for p in self.params]
        self._seen_masks: list[np.ndarray | None] = [
            None if p.mask is None else p.mask.copy() for p in self.params
        ]
        self.revived_last_step: int = 0

    # ------------------------------------------------------------------ subclass hooks
    def _init_state(self, p: Parameter) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def _reset_state(self, state: dict[str, np.ndarray], where: np.ndarray) -> None:
        for buf in state.values():
            buf[where] = 0

    def _update(self, p: Parameter, g: np.ndarray, state: dict, idx) -> None:
        """Update ``p.data[idx]`` from ``g[idx]``; ``idx`` is ``...`` or a boolean mask."""
        raise NotImplementedError

    # ------------------------------------------------------------------ public API
    def step(self) -> None:
        self.revived_last_step = 0
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            state = self.state[i]
            if p.mask is None:
                self._update(p, p.grad, state, ...)
                continue

            seen = self._seen_masks[i]
            revived = p.mask & ~seen
            if revived.any():
                self._reset_state(state, revived)
                self.revived_last_step += int(revived.sum())
            self._seen_masks[i] = p.mask.copy()

            idx = ... if p.mask.all() else p.mask
            self._update(p, p.grad, state, idx)
            p.data[~p.mask] = 0.0  # pruned entries are exact zeros, always

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
