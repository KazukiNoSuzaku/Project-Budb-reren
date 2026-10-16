"""Result persistence and plotting."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: plots are files, never windows
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


def write_csv(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else v


def _parse(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def plot_learning_curve(history: Sequence[dict], path: Path, title: str) -> None:
    epochs = [r["epoch"] for r in history]
    train = [r for r in history if r["epoch"] > 0]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax_loss.plot([r["epoch"] for r in train], [r["train_loss"] for r in train], label="train")
    ax_loss.plot(epochs, [r["val_loss"] for r in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy", title="loss", yscale="log")
    ax_acc.plot([r["epoch"] for r in train], [r["train_acc"] for r in train], label="train")
    ax_acc.plot(epochs, [r["val_acc"] for r in history], label="validation")
    ax_acc.set(
        xlabel="epoch",
        ylabel="accuracy",
        title="accuracy",
        ylim=(max(0.9, min(r["val_acc"] for r in train) - 0.01), 1.0),
    )
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
