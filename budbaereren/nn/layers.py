"""Composable layers."""

from __future__ import annotations

import numpy as np

from budbaereren.engine import Tensor, is_grad_enabled
from budbaereren.nn.init import INITIALIZERS
from budbaereren.nn.module import Module, Parameter


class Linear(Module):
    """``y = x @ (W ⊙ M) + b`` with a prunable weight.

    The forward pass always multiplies by the mask, so a pruned connection is
    removed from the computation even if its storage were somehow non-zero
    (``Parameter.set_mask`` also writes an exact 0.0 there). By the chain rule
    ``∂L/∂W = ∂L/∂(W⊙M) ⊙ M``: the parameter gradient of a pruned entry is
    exactly zero. The *effective* weight is kept with ``retain_grad`` so the
    dense gradient ``∂L/∂(W⊙M)``, i.e. how much the loss would change if a pruned
    connection existed, is available for regrowth.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        init: str = "he",
        dtype=np.float32,
    ):
        self.in_features, self.out_features = in_features, out_features
        w = INITIALIZERS[init](in_features, out_features, rng, dtype=dtype)
        self.weight = Parameter(w, prunable=True, name="weight")
        self.bias = Parameter(np.zeros(out_features, dtype=dtype), name="bias")
        self.effective_weight: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        mask = Tensor(self.weight.mask.astype(self.weight.dtype))
        w_eff = self.weight * mask
        if is_grad_enabled():
            w_eff.retain_grad()
        self.effective_weight = w_eff
        return x @ w_eff + self.bias

    @property
    def dense_grad(self) -> np.ndarray | None:
        """∂L/∂(W⊙M) from the last backward pass, defined at pruned positions too."""
        return None if self.effective_weight is None else self.effective_weight.grad

    def __repr__(self) -> str:
        return f"Linear({self.in_features}, {self.out_features})"


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "gelu": GELU}


class Sequential(Module):
    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, i: int) -> Module:
        return self.layers[i]

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        return "Sequential(\n  " + ",\n  ".join(map(repr, self.layers)) + "\n)"
