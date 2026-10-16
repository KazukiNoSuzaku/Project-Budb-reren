"""Multi-layer perceptron built from composable layers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from budbaereren.engine import Tensor
from budbaereren.nn.layers import ACTIVATIONS, Linear, Sequential
from budbaereren.nn.module import Module


class MLP(Module):
    """``Linear -> act -> ... -> Linear`` (no activation after the last layer).

    Hidden layers use He init (they feed a ReLU-family activation); the output
    layer feeds softmax and uses Xavier init. See ``nn/init.py``.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        rng: np.random.Generator,
        activation: str = "relu",
        dtype=np.float32,
    ):
        if len(sizes) < 2:
            raise ValueError("an MLP needs at least an input and an output size")
        self.sizes = tuple(sizes)
        self.activation = activation
        layers = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
            is_last = i == len(sizes) - 2
            layers.append(
                Linear(fan_in, fan_out, rng, init="xavier" if is_last else "he", dtype=dtype)
            )
            if not is_last:
                layers.append(ACTIVATIONS[activation]())
        self.net = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @property
    def linears(self) -> list[Linear]:
        return [m for m in self.net.layers if isinstance(m, Linear)]

    def __repr__(self) -> str:
        return f"MLP({'-'.join(map(str, self.sizes))}, activation={self.activation})"
