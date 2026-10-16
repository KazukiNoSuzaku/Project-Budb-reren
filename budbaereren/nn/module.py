"""Module and Parameter base classes."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from budbaereren.engine import Tensor


class Parameter(Tensor):
    """A trainable leaf tensor, optionally carrying a pruning mask.

    ``mask`` is the single source of truth for which entries are alive:
    ``None`` means "never prunable" (e.g. biases); otherwise a boolean array of
    the same shape where ``False`` marks a pruned (exactly zero) entry. The
    layer reads it in the forward pass, the optimizer reads it in the update,
    and the pruner writes it.
    """

    def __init__(self, data, prunable: bool = False, name: str | None = None):
        super().__init__(data, requires_grad=True, name=name)
        self.mask: np.ndarray | None = np.ones(self.shape, dtype=bool) if prunable else None

    @property
    def prunable(self) -> bool:
        return self.mask is not None

    def set_mask(self, mask: np.ndarray) -> None:
        """Install a new mask and make pruned entries a real, exact zero."""
        if self.mask is None:
            raise ValueError(f"parameter {self.name!r} is not prunable")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"mask shape {mask.shape} != parameter shape {self.shape}")
        self.mask = mask.copy()
        self.data[~self.mask] = 0.0

    def num_active(self) -> int:
        return self.size if self.mask is None else int(self.mask.sum())


class Module:
    """Base class: discovers Parameters and sub-Modules from attributes, in order."""

    training: bool = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for name, value in vars(self).items():
            full = f"{prefix}{name}"
            if isinstance(value, Parameter):
                yield full, value
            elif isinstance(value, Module):
                yield from value.named_parameters(prefix=f"{full}.")
            elif isinstance(value, list | tuple):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(prefix=f"{full}.{i}.")

    def parameters(self) -> list[Parameter]:
        return [p for _, p in self.named_parameters()]

    def prunable_parameters(self) -> list[Parameter]:
        return [p for p in self.parameters() if p.prunable]

    def modules(self) -> Iterator[Module]:
        yield self
        for value in vars(self).values():
            if isinstance(value, Module):
                yield from value.modules()
            elif isinstance(value, list | tuple):
                for item in value:
                    if isinstance(item, Module):
                        yield from item.modules()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> Module:
        for m in self.modules():
            m.training = mode
        return self

    def eval(self) -> Module:
        return self.train(False)

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        for name, p in self.named_parameters():
            state[name] = p.data.copy()
            if p.mask is not None:
                state[f"{name}.mask"] = p.mask.copy()
        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        for name, p in self.named_parameters():
            p.data[...] = state[name]
            if p.mask is not None and f"{name}.mask" in state:
                p.set_mask(state[f"{name}.mask"])
