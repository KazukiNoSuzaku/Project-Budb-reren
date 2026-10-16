"""Training loop behaviour on a small synthetic problem (no download needed)."""

import numpy as np
import pytest

from budbaereren.nn import MLP
from budbaereren.optim import Adam
from budbaereren.train.data import DEFAULT_CACHE, MNIST, Split, iterate_minibatches, load_mnist
from budbaereren.train.loop import cosine_lr, evaluate, train


def blobs(seed=0, n=600, dim=20, classes=4):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((classes, dim)) * 3
    y = rng.integers(0, classes, n)
    x = (centers[y] + rng.standard_normal((n, dim))).astype(np.float32)
    split = lambda a, b: Split(x[a:b], y[a:b])  # noqa: E731
    return MNIST(train=split(0, 400), val=split(400, 500), test=split(500, 600))


def run(seed=0, epochs=5, **kwargs):
    data = blobs()
    rng = np.random.default_rng(seed)
    model = MLP([20, 16, 4], rng)
    opt = Adam(model.parameters(), lr=1e-2)
    history = train(model, opt, data, epochs, 32, rng, log=None, **kwargs)
    return model, data, history


def test_training_reduces_loss_and_learns():
    model, data, history = run()
    assert history[-1]["val_loss"] < history[0]["val_loss"] / 3
    assert evaluate(model, data.test)[1] > 0.95


def test_training_is_bit_reproducible_for_a_fixed_seed():
    _, _, a = run(seed=3)
    _, _, b = run(seed=3)
    strip = lambda h: [{k: v for k, v in r.items() if k != "seconds"} for r in h]  # noqa: E731
    np.testing.assert_equal(strip(a), strip(b))  # treats the epoch-0 NaN train loss as equal


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nan_guard_raises_immediately():
    data = blobs()
    rng = np.random.default_rng(0)
    model = MLP([20, 16, 4], rng)
    model.parameters()[0].data[0, 0] = np.inf
    with pytest.raises(FloatingPointError, match="step 0"):
        train(model, Adam(model.parameters()), data, 1, 32, rng, log=None)


def test_callbacks_are_invoked():
    steps, epochs = [], []
    run(epochs=2, step_callback=steps.append, epoch_callback=lambda e: epochs.append(e) or {})
    assert steps == list(range(1, 2 * 13 + 1))  # 400 samples / 32 = 13 batches per epoch
    assert epochs == [0, 1, 2]


def test_minibatches_cover_every_example_once_per_epoch():
    split = blobs().train
    seen = np.concatenate([y for _, y in iterate_minibatches(split, 32, np.random.default_rng(0))])
    assert sorted(seen.tolist()) == sorted(split.y.tolist())


def test_cosine_schedule_endpoints():
    s = cosine_lr(1.0, 100, final_frac=0.01)
    assert s(0) == pytest.approx(1.0) and s(100) == pytest.approx(0.01) and s(500) == s(100)


@pytest.mark.skipif(not DEFAULT_CACHE.exists(), reason="MNIST not downloaded yet")
def test_mnist_splits_and_standardisation():
    data = load_mnist()
    assert (len(data.train), len(data.val), len(data.test)) == (55_000, 5_000, 10_000)
    assert abs(data.train.x.mean()) < 1e-3 and abs(data.train.x.std() - 1) < 1e-3
    assert data.test.y[:5].tolist() == [7, 2, 1, 0, 4]  # canonical MNIST test order
