"""MNIST loading.

scikit-learn is used here, and only here, to download the standard MNIST
dataset from OpenML. It is not used for preprocessing, models, training,
metrics or gradients. The raw arrays are cached to ``data/mnist.npz`` so later
runs are offline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "mnist.npz"
N_TRAIN, N_VAL = 55_000, 5_000  # OpenML order: 60k train then 10k test
# SHA-256 of x.tobytes() + y.tobytes() (uint8 pixels, int64 labels). Guarantees that a
# clean clone trains on byte-identical data, whatever OpenML/scikit-learn versions do.
MNIST_SHA256 = "b3fe36794a3a345d09632e9d79f122d61ca91a07e5a22f8536973655bacf2a78"


@dataclass
class Split:
    x: np.ndarray  # (N, 784) float32, standardised
    y: np.ndarray  # (N,) int64

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class MNIST:
    train: Split
    val: Split
    test: Split


def _download_raw(cache: Path) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml  # the ONLY scikit-learn usage in the project

    x, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff", return_X_y=True)
    x = x.astype(np.uint8)
    y = y.astype(np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y)
    return x, y


def load_raw(cache: Path = DEFAULT_CACHE) -> tuple[np.ndarray, np.ndarray]:
    """Return raw (70000, 784) uint8 pixels and (70000,) labels, downloading once."""
    if cache.exists():
        with np.load(cache) as f:
            x, y = f["x"], f["y"]
    else:
        x, y = _download_raw(cache)
    digest = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    if digest != MNIST_SHA256:
        raise ValueError(f"MNIST checksum mismatch ({digest}); delete {cache} and retry")
    return x, y


def load_mnist(cache: Path = DEFAULT_CACHE, subset: int | None = None) -> MNIST:
    """Standard split: 55k train / 5k validation / 10k test, standardised on train.

    ``subset`` (optional) keeps only the first ``subset`` training examples
    for quick smoke runs; validation and test are never subsampled.
    """
    x, y = load_raw(cache)
    if x.shape != (70_000, 784):
        raise ValueError(f"unexpected MNIST shape {x.shape}")
    x = x.astype(np.float32) / 255.0

    x_train, y_train = x[:N_TRAIN], y[:N_TRAIN]
    x_val, y_val = x[N_TRAIN : N_TRAIN + N_VAL], y[N_TRAIN : N_TRAIN + N_VAL]
    x_test, y_test = x[60_000:], y[60_000:]

    # Statistics from the training split only: no information from val/test leaks in.
    mean, std = x_train.mean(), x_train.std()

    def norm(a):
        return ((a - mean) / std).astype(np.float32)

    if subset is not None:
        x_train, y_train = x_train[:subset], y_train[:subset]
    return MNIST(
        train=Split(norm(x_train), y_train),
        val=Split(norm(x_val), y_val),
        test=Split(norm(x_test), y_test),
    )


def iterate_minibatches(split: Split, batch_size: int, rng: np.random.Generator, shuffle=True):
    """Yield (x, y) mini-batches; the permutation comes from ``rng`` (reproducible)."""
    n = len(split)
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield split.x[idx], split.y[idx]
