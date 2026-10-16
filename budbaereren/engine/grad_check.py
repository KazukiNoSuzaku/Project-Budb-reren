"""Finite-difference gradient checking.

Compares the analytic gradients from ``Tensor.backward`` with central
differences ``(f(x + h) - f(x - h)) / 2h``, whose truncation error is O(h^2)
and round-off error is O(eps / h); h = 1e-5 ~ eps^(1/3) balances the two.
Always run in float64: in float32 the rounding error of f (~1e-7 relative)
divided by 2h swamps the comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from budbaereren.engine.tensor import Tensor


def numerical_grad(f: Callable[[], Tensor], x: Tensor, h: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of scalar ``f()`` with respect to ``x.data`` (in place)."""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = x.data[idx]
        x.data[idx] = original + h
        f_plus = f().item()
        x.data[idx] = original - h
        f_minus = f().item()
        x.data[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def relative_error(analytic: np.ndarray, numeric: np.ndarray, floor: float = 1e-12) -> float:
    """Norm-wise relative error ||a - n|| / max(||a|| + ||n||, floor).

    Element-wise relative error is the wrong metric: wherever the true
    derivative crosses zero (e.g. GELU near x = -0.75) the denominator vanishes
    and pure round-off noise in the finite difference reads as a huge error.
    The norm-wise version is scale-invariant and robust to such entries.
    """
    diff = np.linalg.norm((analytic - numeric).ravel())
    return float(
        diff / max(np.linalg.norm(analytic.ravel()) + np.linalg.norm(numeric.ravel()), floor)
    )


def check_gradients(
    f: Callable[..., Tensor],
    inputs: Sequence[np.ndarray],
    h: float = 1e-5,
    seed: int = 0,
) -> list[float]:
    """Check ``f(*tensors)`` against finite differences for every input.

    If ``f`` returns a non-scalar, we check the scalar ``sum(f(...) * R)`` for a
    fixed random ``R``. A random projection exercises every entry of the
    Jacobian, whereas a plain ``sum()`` would only test its column sums.

    Returns the relative error for each input.
    """
    tensors = [Tensor(np.array(x, dtype=np.float64), requires_grad=True) for x in inputs]
    probe = f(*tensors)
    projection = None
    if probe.size != 1:
        projection = np.random.default_rng(seed).standard_normal(probe.shape)

    def scalar_f() -> Tensor:
        out = f(*tensors)
        return (out * projection).sum() if projection is not None else out

    for t in tensors:
        t.zero_grad()
    scalar_f().backward()

    errors = []
    for t in tensors:
        analytic = t.grad if t.grad is not None else np.zeros_like(t.data)
        errors.append(relative_error(analytic, numerical_grad(scalar_f, t, h)))
    return errors
