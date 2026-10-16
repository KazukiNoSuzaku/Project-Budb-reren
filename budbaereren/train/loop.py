"""Mini-batch training loop."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from budbaereren.engine import Tensor, no_grad, ops
from budbaereren.nn import Module
from budbaereren.optim import Optimizer
from budbaereren.train.data import MNIST, Split, iterate_minibatches


def evaluate(model: Module, split: Split, batch_size: int = 5000) -> tuple[float, float]:
    """Mean cross-entropy and accuracy on ``split``, without building a graph."""
    model.eval()
    total_loss, correct = 0.0, 0
    with no_grad():
        for start in range(0, len(split), batch_size):
            x, y = split.x[start : start + batch_size], split.y[start : start + batch_size]
            logits = model(Tensor(x))
            total_loss += ops.softmax_cross_entropy(logits, y).item() * len(y)
            correct += int((logits.data.argmax(axis=1) == y).sum())
    model.train()
    return total_loss / len(split), correct / len(split)


def train(
    model: Module,
    optimizer: Optimizer,
    data: MNIST,
    epochs: int,
    batch_size: int,
    rng: np.random.Generator,
    step_callback: Callable[[int], None] | None = None,
    epoch_callback: Callable[[int], dict] | None = None,
    lr_schedule: Callable[[int], float] | None = None,
    log: Callable[[str], None] | None = print,
) -> list[dict]:
    """Train ``model`` and return one history row per epoch (plus a row for epoch 0).

    ``step_callback(step)`` runs after every optimizer step; it is the hook the
    pruner uses. ``epoch_callback(epoch)`` may return extra metrics to log.
    Raises ``FloatingPointError`` the moment the loss stops being finite.
    """
    history: list[dict] = []
    val_loss, val_acc = evaluate(model, data.val)
    row0 = {
        "epoch": 0,
        "step": 0,
        "train_loss": float("nan"),
        "train_acc": float("nan"),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "seconds": 0.0,
    }
    if epoch_callback is not None:
        row0.update(epoch_callback(0))
    history.append(row0)
    if log:
        log(f"epoch  0 | val loss {val_loss:.4f} acc {val_acc:.4f} (initialisation)")

    step, t0 = 0, time.perf_counter()
    for epoch in range(1, epochs + 1):
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in iterate_minibatches(data.train, batch_size, rng):
            if lr_schedule is not None:
                optimizer.lr = lr_schedule(step)
            optimizer.zero_grad()
            logits = model(Tensor(x))
            loss = ops.softmax_cross_entropy(logits, y)
            if not np.isfinite(loss.item()):
                raise FloatingPointError(f"non-finite loss {loss.item()} at step {step}")
            loss.backward()
            optimizer.step()
            step += 1
            if step_callback is not None:
                step_callback(step)

            loss_sum += loss.item() * len(y)
            correct += int((logits.data.argmax(axis=1) == y).sum())
            seen += len(y)

        for p in model.parameters():
            if not np.all(np.isfinite(p.data)):
                raise FloatingPointError(f"non-finite parameter {p.name} after epoch {epoch}")

        val_loss, val_acc = evaluate(model, data.val)
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "seconds": time.perf_counter() - t0,
        }
        if epoch_callback is not None:
            row.update(epoch_callback(epoch))
        history.append(row)
        if log:
            extra = "".join(
                f" | {k} {v:.4f}" if isinstance(v, float) else f" | {k} {v}"
                for k, v in row.items()
                if k
                not in {
                    "epoch",
                    "step",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "seconds",
                }
            )
            log(
                f"epoch {epoch:2d} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f}"
                f" | val loss {val_loss:.4f} acc {val_acc:.4f} | {row['seconds']:.0f}s{extra}"
            )
    return history


def cosine_lr(base_lr: float, total_steps: int, final_frac: float = 0.01) -> Callable[[int], float]:
    """Cosine decay from ``base_lr`` to ``final_frac * base_lr`` over ``total_steps``.

    A decaying rate lets Adam settle into a minimum instead of bouncing around
    it, which makes the *final* accuracy far less noisy between epochs and seeds.
    That matters more here than the last 0.1% of accuracy: Part 4 compares
    methods across seeds, and epoch-to-epoch jitter would be indistinguishable
    from a real difference.
    """

    def schedule(step: int) -> float:
        progress = min(step / max(total_steps, 1), 1.0)
        return base_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + np.cos(np.pi * progress)))

    return schedule
