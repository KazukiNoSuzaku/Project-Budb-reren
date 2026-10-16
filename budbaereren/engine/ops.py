"""Differentiable operations.

Every op follows the same recipe:

1. compute the forward result with NumPy,
2. define ``backward(g)`` returning one gradient per parent (or ``None``),
3. return ``Tensor._from_op(result, parents, backward, name)``.

Adding a new op means writing one such function; nothing else in the engine
needs to change.
"""

from __future__ import annotations

import numpy as np

from budbaereren.engine.tensor import Tensor


# ---------------------------------------------------------------------- helpers
def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` by undoing NumPy broadcasting.

    Broadcasting copies an operand along (a) new leading axes and (b) axes of
    size 1. The adjoint of "copy along an axis" is "sum along that axis", so we
    sum over exactly those axes.
    """
    if grad.shape == shape:
        return grad
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad.reshape(shape)


def as_tensor(x, like: Tensor | None = None) -> Tensor:
    """Wrap ``x`` as a constant Tensor.

    Python scalars take the dtype of ``like`` so that ``float32_tensor * 2.0``
    stays float32 (a 0-d float64 array would otherwise upcast the result).
    """
    if isinstance(x, Tensor):
        return x
    if like is not None and isinstance(x, int | float):
        return Tensor(np.asarray(x, dtype=like.data.dtype))
    return Tensor(x)


# ---------------------------------------------------------------------- element-wise binary
def add(a, b) -> Tensor:
    a, b = as_tensor(a, b if isinstance(b, Tensor) else None), as_tensor(b, a)

    def backward(g):
        return unbroadcast(g, a.shape), unbroadcast(g, b.shape)

    return Tensor._from_op(a.data + b.data, (a, b), backward, "add")


def sub(a, b) -> Tensor:
    a, b = as_tensor(a, b if isinstance(b, Tensor) else None), as_tensor(b, a)

    def backward(g):
        return unbroadcast(g, a.shape), unbroadcast(-g, b.shape)

    return Tensor._from_op(a.data - b.data, (a, b), backward, "sub")


def mul(a, b) -> Tensor:
    a, b = as_tensor(a, b if isinstance(b, Tensor) else None), as_tensor(b, a)

    def backward(g):
        return unbroadcast(g * b.data, a.shape), unbroadcast(g * a.data, b.shape)

    return Tensor._from_op(a.data * b.data, (a, b), backward, "mul")


def div(a, b) -> Tensor:
    a, b = as_tensor(a, b if isinstance(b, Tensor) else None), as_tensor(b, a)

    def backward(g):
        ga = g / b.data
        gb = -g * a.data / (b.data * b.data)
        return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

    return Tensor._from_op(a.data / b.data, (a, b), backward, "div")


def neg(a: Tensor) -> Tensor:
    return Tensor._from_op(-a.data, (a,), lambda g: (-g,), "neg")


# ---------------------------------------------------------------------- operator overloading
Tensor.__add__ = add
Tensor.__radd__ = lambda self, other: add(other, self)
Tensor.__sub__ = sub
Tensor.__rsub__ = lambda self, other: sub(other, self)
Tensor.__mul__ = mul
Tensor.__rmul__ = lambda self, other: mul(other, self)
Tensor.__truediv__ = div
Tensor.__rtruediv__ = lambda self, other: div(other, self)
Tensor.__neg__ = neg
