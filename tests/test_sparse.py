"""Sparse inference: CSR correctness, compaction exactness, honest cost accounting."""

import numpy as np
import pytest

from budbaereren.engine import Tensor, no_grad
from budbaereren.nn import MLP
from budbaereren.sparse import (
    CompactMLP,
    CSRMatrix,
    DenseMLP,
    SparseMLP,
    compact_flops,
    dense_flops,
    sparse_bytes,
    sparse_flops,
)


def random_sparse(shape, density, rng):
    w = rng.standard_normal(shape).astype(np.float32)
    w[rng.random(shape) >= density] = 0.0
    return w


@pytest.mark.parametrize("density", [1.0, 0.5, 0.1, 0.01, 0.0])
def test_csr_roundtrip_and_matmul(density):
    rng = np.random.default_rng(0)
    w = random_sparse((40, 30), density, rng)
    csr = CSRMatrix.from_dense_weight(w)
    assert csr.nnz == np.count_nonzero(w)
    np.testing.assert_array_equal(csr.to_dense_weight(), w)
    x = rng.standard_normal((7, 40)).astype(np.float32)
    np.testing.assert_allclose(csr.matmul_from_left(x), x @ w, rtol=1e-5, atol=1e-5)


def test_csr_handles_empty_rows_between_nonempty_ones():
    """np.add.reduceat mis-sums empty segments; the implementation must skip them."""
    w = np.zeros((4, 5), dtype=np.float32)
    w[:, 0] = [1, 2, 3, 4]
    w[1, 3] = 5.0
    w[2, 4] = -1.0  # output columns 1 and 2 are entirely empty
    csr = CSRMatrix.from_dense_weight(w)
    x = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.testing.assert_allclose(csr.matmul_from_left(x), x @ w)


def pruned_model(rng, density=0.1):
    model = MLP([20, 32, 16, 4], rng)
    for lin in model.linears:
        mask = rng.random(lin.weight.shape) < density
        lin.weight.set_mask(mask)
        lin.bias.data[...] = rng.standard_normal(lin.bias.shape)
    return model


def test_all_forward_paths_agree_with_the_training_model():
    rng = np.random.default_rng(1)
    model = pruned_model(rng, density=0.15)
    x = rng.standard_normal((64, 20)).astype(np.float32)
    with no_grad():
        reference = model(Tensor(x)).data
    dense = DenseMLP.from_model(model)
    for path in (dense, SparseMLP.from_dense(dense), CompactMLP.from_dense(dense)):
        np.testing.assert_allclose(path.forward(x), reference, rtol=1e-4, atol=1e-5)


def test_compaction_removes_disconnected_units_and_folds_constant_ones():
    rng = np.random.default_rng(2)
    model = MLP([6, 5, 3], rng)
    w1, w2 = model.linears[0].weight, model.linears[1].weight
    m1, m2 = np.ones(w1.shape, bool), np.ones(w2.shape, bool)
    m1[0, :] = False  # input 0 feeds nothing          -> input dropped
    m2[1, :] = False  # hidden 1 feeds nothing          -> unit dropped
    m1[:, 2] = False  # hidden 2 has no inputs: constant relu(b2) -> folded, dropped
    w1.set_mask(m1)
    w2.set_mask(m2)
    model.linears[0].bias.data[2] = 0.7  # make the folded constant non-trivial
    dense = DenseMLP.from_model(model)
    compact = CompactMLP.from_dense(dense)
    assert compact.sizes == [5, 3, 3]
    x = rng.standard_normal((10, 6)).astype(np.float32)
    np.testing.assert_allclose(compact.forward(x), dense.forward(x), rtol=1e-5, atol=1e-6)


def test_flop_accounting():
    rng = np.random.default_rng(3)
    model = pruned_model(rng, density=0.2)
    dense = DenseMLP.from_model(model)
    sparse = SparseMLP.from_dense(dense)
    nnz = [np.count_nonzero(w) for w in dense.weights]
    biases_and_relu = (32 + 32) + (16 + 16) + 4
    assert dense_flops(dense) == 2 * (20 * 32 + 32 * 16 + 16 * 4) + biases_and_relu
    assert sparse_flops(sparse) == 2 * sum(nnz) + biases_and_relu
    assert compact_flops(CompactMLP.from_dense(dense)) <= dense_flops(dense)
    # CSR storage: 4 bytes value + 4 bytes index per nnz, 8 bytes per row pointer, + biases
    expected = sum(8 * n for n in nnz) + 8 * (33 + 17 + 5) + 4 * (32 + 16 + 4)
    assert sparse_bytes(sparse) == expected
