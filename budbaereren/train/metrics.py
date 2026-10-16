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


def plot_pruning_run(history_csv: Path, events_csv: Path, path: Path, title: str) -> None:
    """Sparsity schedule (prune events) and validation accuracy on a shared epoch axis."""
    history = read_csv(history_csv)
    events = read_csv(events_csv) if events_csv.exists() and events_csv.stat().st_size else []
    steps_per_epoch = history[-1]["step"] / history[-1]["epoch"]

    fig, ax_s = plt.subplots(figsize=(8, 4))
    if events:
        # Sparsity holds after the last event, so extend the step line to the final epoch.
        ev_epochs = [0.0] + [e["step"] / steps_per_epoch for e in events] + [history[-1]["epoch"]]
        sparsity = [0.0] + [e["achieved"] for e in events] + [events[-1]["achieved"]]
        ax_s.step(
            ev_epochs, sparsity, where="post", color="tab:gray", label="sparsity (prune events)"
        )
    ax_s.set(xlabel="epoch", ylabel="sparsity", ylim=(0, 1))
    ax_a = ax_s.twinx()
    ax_a.plot(
        [r["epoch"] for r in history],
        [r["val_acc"] for r in history],
        "o-",
        color="tab:blue",
        ms=3,
        label="validation accuracy",
    )
    ax_a.set(ylabel="validation accuracy", ylim=(0.9, 1.0))
    handles = ax_s.get_legend_handles_labels()[0] + ax_a.get_legend_handles_labels()[0]
    ax_s.legend(handles, [h.get_label() for h in handles], loc="lower right")
    ax_s.grid(alpha=0.3)
    ax_s.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_s.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_regrowth(regrowth_csv: Path, path: Path, title: str) -> None:
    """Swap fraction and resulting mask churn (1 − Jaccard) per regrowth event."""
    events = read_csv(regrowth_csv)
    steps = [e["step"] for e in events]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(steps, [e["fraction"] for e in events], label="scheduled swap fraction α_t")
    ax.plot(
        steps, [1 - e["jaccard"] for e in events], "o", ms=3, label="measured churn (1 − Jaccard)"
    )
    ax.set(xlabel="training step", ylabel="fraction of active weights", title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
