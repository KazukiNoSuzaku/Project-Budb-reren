"""Weight initialisation schemes.

He / Kaiming normal (He et al., 2015) for layers followed by ReLU:
    Var[W] = 2 / fan_in.
A ReLU zeroes (in expectation) half of its pre-activations, halving the
second moment that reaches the next layer; the factor 2 compensates, so the
activation scale stays constant with depth instead of shrinking
geometrically (vanishing signal) or growing (exploding).

Xavier / Glorot uniform (Glorot & Bengio, 2010) for the output layer:
    W ~ U(-a, a),  a = sqrt(6 / (fan_in + fan_out))  =>  Var[W] = 2 / (fan_in + fan_out).
It balances forward and backward variance for a linear (pre-softmax) layer,
keeping initial logits O(1) so the initial loss is close to log(C).
"""

from __future__ import annotations

import numpy as np


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator, dtype=np.float32):
    return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(dtype)


def xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator, dtype=np.float32):
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, (fan_in, fan_out)).astype(dtype)


INITIALIZERS = {"he": he_normal, "xavier": xavier_uniform}
