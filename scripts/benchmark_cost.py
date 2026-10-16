"""Part 4.2: measure what pruning actually saves.

For Fisher-pruned models (seed 0) at several sparsity levels, compare three
forward paths over identical weights:

  dense    x @ (W*M) + b on full matrices. Multiplies by stored zeros: NOT a speedup.
  csr      hand-written CSR, work proportional to nnz.
  compact  dense BLAS on the sub-network left after deleting disconnected units.

Reports FLOPs/example, parameter bytes, active parameters, test accuracy, max
logit difference between paths, and median wall-clock latency at batch 1 and
batch 256.

Usage:  uv run python scripts/benchmark_cost.py
        (models are trained once and cached in results/part4/models/)
Writes: results/part4/cost.csv, results/part4/cost.png, results/part4/cost_env.json
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np

from budbaereren.nn import MLP
from budbaereren.sparse import (
    CompactMLP,
    DenseMLP,
    SparseMLP,
    compact_bytes,
    compact_flops,
    dense_bytes,
    dense_flops,
    sparse_bytes,
    sparse_flops,
)
from budbaereren.train.data import load_mnist
from budbaereren.train.experiment import RunConfig, run
from budbaereren.train.metrics import plot_cost, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "part4"
LEVELS = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99]
SIZES = [784, 300, 100, 10]


def get_model(level: float, data) -> MLP:
    path = OUT / "models" / f"fisher_s{round(level * 100)}_seed0.npz"
    model = MLP(SIZES, np.random.default_rng(0))
    if not path.exists():
        print(f"training seed-0 Fisher model at {level:.0%} sparsity ...", flush=True)
        _, trained = run(RunConfig(criterion="fisher", target_sparsity=level, seed=0), data)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **trained.state_dict())
    with np.load(path) as f:
        model.load_state_dict(dict(f))
    return model


def median_latency(fn, x, repeats: int) -> float:
    for _ in range(max(3, repeats // 10)):
        fn(x)  # warm-up: caches, allocator, lazy initialisation
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    data = load_mnist()
    x_test, y_test = data.test.x, data.test.y
    rng = np.random.default_rng(0)
    batch1 = x_test[rng.integers(0, len(x_test), 1)]
    batch256 = x_test[rng.integers(0, len(x_test), 256)]

    rows = []
    for level in LEVELS:
        model = get_model(level, data)
        dense = DenseMLP.from_model(model)
        paths = {
            "dense": dense,
            "csr": SparseMLP.from_dense(dense),
            "compact": CompactMLP.from_dense(dense),
        }
        cost = {
            "dense": (dense_flops(dense), dense_bytes(dense)),
            "csr": (sparse_flops(paths["csr"]), sparse_bytes(paths["csr"])),
            "compact": (compact_flops(paths["compact"]), compact_bytes(paths["compact"])),
        }
        reference = dense.forward(x_test)
        active = sum(int(np.count_nonzero(w)) for w in dense.weights)
        active += sum(b.size for b in dense.biases)
        for name, path in paths.items():
            logits = path.forward(x_test)
            row = {
                "target_sparsity": level,
                "path": name,
                "active_params": active,
                "flops_per_example": cost[name][0],
                "param_bytes": cost[name][1],
                "test_acc": float((logits.argmax(1) == y_test).mean()),
                "max_abs_logit_diff_vs_dense": float(np.abs(logits - reference).max()),
                "latency_b1_us": 1e6 * median_latency(path.forward, batch1, 2000),
                "latency_b256_us": 1e6 * median_latency(path.forward, batch256, 300),
                "layer_sizes": "-".join(map(str, path.sizes)) if name == "compact" else "",
            }
            rows.append(row)
            print(
                f"{level:5.0%} {name:8s} FLOPs {row['flops_per_example']:>9,}  "
                f"b1 {row['latency_b1_us']:8.1f} us  b256 {row['latency_b256_us']:9.1f} us  "
                f"acc {row['test_acc']:.4f}  {row['layer_sizes']}",
                flush=True,
            )

    write_csv(rows, OUT / "cost.csv")
    plot_cost(OUT / "cost.csv", OUT / "cost.png")
    write_json(
        {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blas": "see numpy.show_config(); Accelerate on macOS arm64",
            "thread_env": {
                k: os.environ.get(k)
                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
            "timing": "median of 2000 (batch 1) / 300 (batch 256) runs after warm-up",
        },
        OUT / "cost_env.json",
    )


if __name__ == "__main__":
    main()
