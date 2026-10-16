"""The Tensor type and the reverse-mode backward pass.

A ``Tensor`` wraps a NumPy array. Every differentiable op (see ``ops.py``)
returns a new Tensor that remembers its parents and a ``_backward`` closure:

    _backward(upstream_grad) -> tuple of gradients, one per parent

The closure is pure: it returns gradients and never touches parent state.
``Tensor.backward`` owns the bookkeeping (topological order, accumulation),
so an op cannot get accumulation wrong.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator

import numpy as np

_GRAD_ENABLED = True


@contextlib.contextmanager
def no_grad() -> Iterator[None]:
    """Disable graph construction (for evaluation / inference)."""
    global _GRAD_ENABLED
    previous = _GRAD_ENABLED
    _GRAD_ENABLED = False
    try:
        yield
    finally:
        _GRAD_ENABLED = previous


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


def _to_array(data, dtype=None) -> np.ndarray:
    arr = np.asarray(data, dtype=dtype)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


class Tensor:
    """An n-dimensional array that participates in reverse-mode autodiff."""

    __array_priority__ = 1000  # make `ndarray <op> Tensor` defer to Tensor's reflected ops

    def __init__(self, data, requires_grad: bool = False, dtype=None, name: str | None = None):
        self.data: np.ndarray = _to_array(data, dtype)
        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray | None = None
        self.name = name
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Callable[[np.ndarray], tuple[np.ndarray | None, ...]] | None = None
        self._op: str = ""
        self._retain: bool = False

    # ------------------------------------------------------------------ graph construction
    @classmethod
    def _from_op(
        cls,
        data: np.ndarray,
        parents: tuple[Tensor, ...],
        backward: Callable[[np.ndarray], tuple[np.ndarray | None, ...]],
        op: str,
    ) -> Tensor:
        """Create the output of an op, recording the graph edge only if needed."""
        out = cls(data)
        if _GRAD_ENABLED and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._parents = parents
            out._backward = backward
            out._op = op
        return out

    def retain_grad(self) -> Tensor:
        """Ask backward() to store this (non-leaf) tensor's gradient in ``.grad``."""
        self._retain = True
        return self

    @property
    def is_leaf(self) -> bool:
        return not self._parents

    # ------------------------------------------------------------------ backward pass
    def backward(self, grad: np.ndarray | None = None) -> None:
        """Backpropagate from this tensor through the graph.

        ``grad`` defaults to 1 and is only optional for scalar outputs.
        Leaf gradients *accumulate* across calls; call ``zero_grad`` between steps.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given explicitly for non-scalar outputs")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=self.data.dtype)
        if grad.shape != self.data.shape:
            raise ValueError(f"grad shape {grad.shape} != tensor shape {self.data.shape}")

        grads: dict[int, np.ndarray] = {id(self): grad}
        for node in reversed(_topological_order(self)):
            g = grads.pop(id(node), None)
            if g is None:
                continue  # no path from the output reaches this node with a gradient
            if node.is_leaf:
                node.grad = g.copy() if node.grad is None else node.grad + g
            elif node._retain:
                node.grad = g.copy()  # intermediates are overwritten, never accumulated
            if node._backward is None:
                continue
            parent_grads = node._backward(g)
            for parent, pg in zip(node._parents, parent_grads, strict=True):
                if pg is None or not parent.requires_grad:
                    continue
                if pg.shape != parent.data.shape:
                    raise RuntimeError(
                        f"op '{node._op}' produced grad of shape {pg.shape} "
                        f"for parent of shape {parent.data.shape}"
                    )
                key = id(parent)
                grads[key] = grads[key] + pg if key in grads else pg

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------------------ conveniences
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return self.data.item()

    def detach(self) -> Tensor:
        return Tensor(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        grad = ", requires_grad=True" if self.requires_grad else ""
        op = f", op='{self._op}'" if self._op else ""
        return f"Tensor({self.data!r}{grad}{op})"


def _topological_order(root: Tensor) -> list[Tensor]:
    """Return nodes so that every node appears after all of its parents.

    Iterative post-order DFS: an explicit stack avoids Python's recursion
    limit on deep graphs. A node is emitted only after all of its parents
    have been emitted, which is exactly the order backward() must reverse.
    """
    order: list[Tensor] = []
    visited: set[int] = set()
    stack: list[tuple[Tensor, bool]] = [(root, False)]
    while stack:
        node, parents_done = stack.pop()
        if parents_done:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for parent in node._parents:
            if id(parent) not in visited:
                stack.append((parent, False))
    return order
