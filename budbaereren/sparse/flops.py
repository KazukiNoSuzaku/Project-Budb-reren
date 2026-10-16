"""Analytic cost model: FLOPs and memory per example for each forward path.

Conventions (stated so the numbers are unambiguous):

* one multiply-accumulate (MAC) = 2 FLOPs;
* a dense layer (in → out) costs 2·in·out FLOPs for the matmul, plus ``out``
  bias adds, plus ``out`` ReLU comparisons for hidden layers;
* a CSR layer costs 2·nnz FLOPs (nnz multiplies + nnz adds) plus the same bias
  and ReLU terms. Pruned connections cost nothing;
* memory counts the bytes of parameters that must be read: dense float32 is
  4·in·out; CSR is 4·nnz (values) + 4·nnz (int32 column indices)
  + 8·(out + 1) (int64 row pointers). The index overhead is why CSR only saves
  memory below 50 % density.

These are hardware-independent; wall-clock time is measured separately.
"""

from __future__ import annotations

from budbaereren.sparse.inference import CompactMLP, DenseMLP, SparseMLP


def dense_flops(model: DenseMLP) -> int:
    total = 0
    for i, w in enumerate(model.weights):
        fan_in, fan_out = w.shape
        total += 2 * fan_in * fan_out + fan_out
        if i < len(model.weights) - 1:
            total += fan_out
    return total


def sparse_flops(model: SparseMLP) -> int:
    total = 0
    for i, layer in enumerate(model.layers):
        fan_out = layer.shape[0]
        total += 2 * layer.nnz + fan_out
        if i < len(model.layers) - 1:
            total += fan_out
    return total


def compact_flops(model: CompactMLP) -> int:
    return dense_flops(DenseMLP(model.weights, model.biases))


def dense_bytes(model: DenseMLP) -> int:
    return sum(w.nbytes + b.nbytes for w, b in zip(model.weights, model.biases, strict=True))


def sparse_bytes(model: SparseMLP) -> int:
    return sum(
        layer.nbytes() + b.nbytes for layer, b in zip(model.layers, model.biases, strict=True)
    )


def compact_bytes(model: CompactMLP) -> int:
    return dense_bytes(DenseMLP(model.weights, model.biases)) + model.input_index.nbytes
