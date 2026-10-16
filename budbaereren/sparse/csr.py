"""A hand-written CSR (compressed sparse row) matrix and sparse-aware forward pass.

For a layer ``y = x @ W + b`` with ``W`` of shape (in, out), we store ``Wᵀ`` in
CSR form: one row per *output* unit, listing the inputs it is connected to.

    data[k]     value of the k-th stored (non-zero) weight
    indices[k]  input index of that weight
    indptr[j]   start of output row j in data/indices (row j is k ∈ [indptr[j], indptr[j+1]))

Then for every output unit j:   y[:, j] = Σ_{k in row j} x[:, indices[k]] · data[k]

The forward pass is vectorised over the whole batch and all non-zeros:
gather the needed input columns, multiply by the weights, and segment-sum each
output row with ``np.add.reduceat``. It does exactly ``nnz`` multiplies and
``nnz`` adds per example; pruned connections are never touched, never
multiplied by zero, and do not occupy memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CSRMatrix:
    data: np.ndarray  # (nnz,)
    indices: np.ndarray  # (nnz,) int32, input index
    indptr: np.ndarray  # (n_rows + 1,) int64
    shape: tuple[int, int]  # (n_rows, n_cols) = (out, in)

    @classmethod
    def from_dense_weight(cls, weight: np.ndarray) -> CSRMatrix:
        """Build CSR of ``weightᵀ`` from a dense (in, out) weight; zeros are dropped."""
        wt = np.ascontiguousarray(weight.T)  # (out, in)
        rows, cols = np.nonzero(wt)  # row-major order: already grouped by row
        indptr = np.zeros(wt.shape[0] + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr)
        return cls(wt[rows, cols].copy(), cols.astype(np.int32), indptr, wt.shape)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def to_dense_weight(self) -> np.ndarray:
        wt = np.zeros(self.shape, dtype=self.data.dtype)
        rows = np.repeat(np.arange(self.shape[0]), np.diff(self.indptr))
        wt[rows, self.indices] = self.data
        return wt.T

    def matmul_from_left(self, x: np.ndarray) -> np.ndarray:
        """Compute ``x @ W`` (= ``(W_csrᵀ applied to rows of x)``) for x of shape (B, in)."""
        n_out = self.shape[0]
        out = np.zeros((x.shape[0], n_out), dtype=np.result_type(x, self.data))
        if self.nnz == 0:
            return out
        products = x[:, self.indices] * self.data  # (B, nnz): exactly nnz multiplies per row
        nonempty = np.flatnonzero(np.diff(self.indptr))
        # reduceat over the start offsets of non-empty rows sums each row's segment.
        out[:, nonempty] = np.add.reduceat(products, self.indptr[nonempty], axis=1)
        return out

    def nbytes(self) -> int:
        return self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
