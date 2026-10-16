"""Inference-only model representations exported from a trained (pruned) MLP.

Three forward paths over the same trained weights, all pure NumPy, no autodiff:

* ``DenseMLP``       x @ W + b with the full (masked) matrices: the reference.
                     Multiplies by every stored zero, which is NOT a speedup.
* ``SparseMLP``      each layer as a hand-written CSR matrix: work ∝ nnz.
* ``CompactMLP``     structured compaction: physically delete input features
                     and hidden units that pruning disconnected, then run dense
                     BLAS on the smaller matrices.

All three must produce the same logits (tested to ~1e-6 in float32).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from budbaereren.sparse.csr import CSRMatrix


def _relu(x):
    return np.maximum(x, 0.0)


@dataclass
class DenseMLP:
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    @classmethod
    def from_model(cls, model) -> DenseMLP:
        ws = [lin.weight.data * lin.weight.mask for lin in model.linears]
        return cls(
            [w.astype(np.float32) for w in ws],
            [lin.bias.data.astype(np.float32) for lin in model.linears],
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = _relu(x)
        return x


@dataclass
class SparseMLP:
    layers: list[CSRMatrix]
    biases: list[np.ndarray]

    @classmethod
    def from_dense(cls, dense: DenseMLP) -> SparseMLP:
        return cls([CSRMatrix.from_dense_weight(w) for w in dense.weights], dense.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.layers, self.biases, strict=True)):
            x = w.matmul_from_left(x) + b
            if i < len(self.layers) - 1:
                x = _relu(x)
        return x


@dataclass
class CompactMLP:
    """Dense MLP on the sub-network that pruning left connected.

    Exact transformations (the logits are unchanged):

    * an input feature with no outgoing weight is never read: drop it;
    * a hidden unit with no outgoing weight cannot affect the output: drop it;
    * a hidden unit with no *incoming* weight outputs the constant relu(b_j):
      fold relu(b_j) · W_next[j, :] into the next layer's bias, then drop it.

    Repeat until nothing changes (removing a unit can disconnect another).
    """

    input_index: np.ndarray  # which original input features are read
    weights: list[np.ndarray]
    biases: list[np.ndarray]

    @classmethod
    def from_dense(cls, dense: DenseMLP) -> CompactMLP:
        ws = [w.copy() for w in dense.weights]
        bs = [b.copy() for b in dense.biases]
        input_index = np.arange(ws[0].shape[0])
        changed = True
        while changed:
            changed = False
            keep_in = np.flatnonzero(np.any(ws[0] != 0, axis=1))
            if keep_in.size < ws[0].shape[0]:
                ws[0], input_index, changed = ws[0][keep_in], input_index[keep_in], True
            for layer in range(len(ws) - 1):  # hidden units between layer and layer+1
                w_in, w_out = ws[layer], ws[layer + 1]
                no_out = ~np.any(w_out != 0, axis=1)
                no_in = ~np.any(w_in != 0, axis=0)
                const = no_in & ~no_out
                if const.any():
                    bs[layer + 1] = bs[layer + 1] + _relu(bs[layer][const]) @ w_out[const]
                drop = no_out | no_in
                if drop.any():
                    keep = ~drop
                    ws[layer], bs[layer] = w_in[:, keep], bs[layer][keep]
                    ws[layer + 1] = w_out[keep]
                    changed = True
        return cls(input_index, ws, bs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x[:, self.input_index]
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = _relu(x)
        return x

    @property
    def sizes(self) -> list[int]:
        return [len(self.input_index)] + [w.shape[1] for w in self.weights]
