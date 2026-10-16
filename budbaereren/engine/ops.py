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


# ---------------------------------------------------------------------- linear algebra
def matmul(a, b) -> Tensor:
    """2-D matrix product. ``(N, D) @ (D, K) -> (N, K)``."""
    a, b = as_tensor(a), as_tensor(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"matmul supports 2-D operands only, got {a.shape} @ {b.shape}")

    def backward(g):
        # out_ik = Σ_j a_ij b_jk  =>  ∂a_ij = Σ_k g_ik b_jk,  ∂b_jk = Σ_i a_ij g_ik
        return g @ b.data.T, a.data.T @ g

    return Tensor._from_op(a.data @ b.data, (a, b), backward, "matmul")


# ---------------------------------------------------------------------- reductions
def _normalize_axes(axis, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return tuple(sorted(ax % ndim for ax in axes))


def _expand_reduced(g: np.ndarray, axes: tuple[int, ...], keepdims: bool, shape) -> np.ndarray:
    """Undo a reduction's shape change so ``g`` broadcasts against the input."""
    if not keepdims:
        g = np.expand_dims(g, axes)
    return np.broadcast_to(g, shape)


def sum(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:  # noqa: A001 - mirrors numpy
    axes = _normalize_axes(axis, a.ndim)

    def backward(g):
        # Every input element contributes with weight 1 to its reduced output.
        return (_expand_reduced(g, axes, keepdims, a.shape).copy(),)

    return Tensor._from_op(a.data.sum(axis=axes, keepdims=keepdims), (a,), backward, "sum")


def mean(a: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    axes = _normalize_axes(axis, a.ndim)
    count = int(np.prod([a.shape[ax] for ax in axes])) if axes else 1

    def backward(g):
        return (_expand_reduced(g, axes, keepdims, a.shape) / count,)

    return Tensor._from_op(a.data.mean(axis=axes, keepdims=keepdims), (a,), backward, "mean")


# ---------------------------------------------------------------------- element-wise unary
def relu(a: Tensor) -> Tensor:
    active = a.data > 0
    # Subgradient choice at exactly 0: 0 (the point is measure-zero for real data).
    return Tensor._from_op(a.data * active, (a,), lambda g: (g * active,), "relu")


def tanh(a: Tensor) -> Tensor:
    t = np.tanh(a.data)
    return Tensor._from_op(t, (a,), lambda g: (g * (1.0 - t * t),), "tanh")


def sigmoid(a: Tensor) -> Tensor:
    # σ(x) = ½(1 + tanh(x/2)) is algebraically identical to 1/(1+e^{-x}) but never overflows.
    s = 0.5 * (1.0 + np.tanh(0.5 * a.data))
    return Tensor._from_op(s, (a,), lambda g: (g * s * (1.0 - s),), "sigmoid")


_GELU_C = np.sqrt(2.0 / np.pi)


def gelu(a: Tensor) -> Tensor:
    """GELU, tanh approximation (Hendrycks & Gimpel, 2016)."""
    x = a.data
    c = np.asarray(_GELU_C, dtype=x.dtype)
    inner = c * (x + 0.044715 * x**3)
    t = np.tanh(inner)

    def backward(g):
        # d/dx [½x(1+t)] = ½(1+t) + ½x(1−t²)·c(1 + 3·0.044715x²)
        d_inner = c * (1.0 + 3 * 0.044715 * x * x)
        return (g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * d_inner),)

    return Tensor._from_op(0.5 * x * (1.0 + t), (a,), backward, "gelu")


def exp(a: Tensor) -> Tensor:
    e = np.exp(a.data)
    return Tensor._from_op(e, (a,), lambda g: (g * e,), "exp")


def log(a: Tensor) -> Tensor:
    return Tensor._from_op(np.log(a.data), (a,), lambda g: (g / a.data,), "log")


def pow(a: Tensor, exponent: float) -> Tensor:  # noqa: A001 - mirrors builtin
    """``a ** exponent`` for a constant scalar exponent."""
    if isinstance(exponent, Tensor):
        raise TypeError("pow supports a constant scalar exponent only")

    def backward(g):
        return (g * exponent * a.data ** (exponent - 1),)

    return Tensor._from_op(a.data**exponent, (a,), backward, "pow")


# ---------------------------------------------------------------------- losses
def softmax_cross_entropy(logits: Tensor, targets) -> Tensor:
    """Mean cross-entropy of ``softmax(logits)`` against integer class ``targets``.

    Fused and numerically stable: uses the log-sum-exp trick with the row max
    subtracted, so no exponent is ever positive. Gradient: (softmax - onehot) / N.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, C), got {logits.shape}")
    y = np.asarray(targets.data if isinstance(targets, Tensor) else targets).astype(np.int64)
    n = logits.shape[0]
    rows = np.arange(n)

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = exp_shifted.sum(axis=1, keepdims=True)
    log_probs_true = shifted[rows, y] - np.log(sum_exp[:, 0])
    loss = -log_probs_true.mean()

    def backward(g):
        grad = exp_shifted / sum_exp  # softmax probabilities
        grad[rows, y] -= 1.0
        return (grad * (g / n),)

    return Tensor._from_op(np.asarray(loss, dtype=logits.dtype), (logits,), backward, "softmax_ce")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax on raw arrays (inference helper, not differentiable)."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


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
Tensor.__matmul__ = matmul
Tensor.__rmatmul__ = lambda self, other: matmul(other, self)
Tensor.__pow__ = pow
Tensor.sum = sum
Tensor.mean = mean
Tensor.relu = relu
Tensor.tanh = tanh
Tensor.sigmoid = sigmoid
Tensor.gelu = gelu
Tensor.exp = exp
Tensor.log = log
