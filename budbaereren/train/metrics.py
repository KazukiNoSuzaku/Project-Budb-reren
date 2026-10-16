"""Result persistence and plotting."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np

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


# ---------------------------------------------------------------------- plot style
# Categorical slots in fixed order from a colour-vision-validated palette (the first
# three slots pass all-pairs CVD separation). Gray is reserved for reference lines.
# Text stays in ink colours; colour carries series identity only. Never a dual axis.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE = "#8a8984"
INK, INK_MUTED = "#0b0b0b", "#52514e"


def _style(ax) -> None:
    ax.grid(color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ae")
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK_MUTED)
    ax.yaxis.label.set_color(INK_MUTED)
    ax.title.set_color(INK)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def plot_learning_curve(history: Sequence[dict], path: Path, title: str) -> None:
    epochs = [r["epoch"] for r in history]
    train = [r for r in history if r["epoch"] > 0]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, name in (
        (ax_loss, "loss", "cross-entropy (log scale)"),
        (ax_acc, "acc", "accuracy"),
    ):
        ax.plot(
            [r["epoch"] for r in train],
            [r[f"train_{key}"] for r in train],
            color=SERIES[0],
            lw=2,
            label="train",
        )
        ax.plot(
            epochs, [r[f"val_{key}"] for r in history], color=SERIES[1], lw=2, label="validation"
        )
        ax.set(xlabel="epoch", ylabel=name)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _style(ax)
        ax.legend(frameon=False, fontsize=9)
    ax_loss.set(yscale="log", title="loss")
    ax_acc.set(title="accuracy", ylim=(max(0.9, min(r["val_acc"] for r in train) - 0.01), 1.0))
    fig.suptitle(title, color=INK)
    _save(fig, path)


def plot_pruning_run(history_csv: Path, events_csv: Path, path: Path, title: str) -> None:
    """Sparsity (prune events) and validation accuracy: two panels, shared epoch axis."""
    history = read_csv(history_csv)
    events = read_csv(events_csv) if events_csv.exists() and events_csv.stat().st_size else []
    steps_per_epoch = history[-1]["step"] / history[-1]["epoch"]
    fig, (ax_s, ax_a) = plt.subplots(
        2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]}
    )
    if events:
        # Sparsity holds after the last event, so extend the step line to the final epoch.
        ev_epochs = [0.0] + [e["step"] / steps_per_epoch for e in events] + [history[-1]["epoch"]]
        sparsity = [0.0] + [100 * e["achieved"] for e in events] + [100 * events[-1]["achieved"]]
        ax_s.step(ev_epochs, sparsity, where="post", color=SERIES[0], lw=2)
    ax_s.set(ylabel="sparsity (%)", ylim=(0, 100), title=title)
    ax_a.plot(
        [r["epoch"] for r in history],
        [100 * r["val_acc"] for r in history],
        "o-",
        color=SERIES[1],
        lw=2,
        ms=4,
    )
    ax_a.set(xlabel="epoch", ylabel="validation accuracy (%)", ylim=(90, 100))
    ax_a.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in (ax_s, ax_a):
        _style(ax)
    _save(fig, path)


def plot_regrowth(regrowth_csv: Path, path: Path, title: str) -> None:
    """Swap fraction and resulting mask churn (1 − Jaccard) per regrowth event."""
    events = read_csv(regrowth_csv)
    steps = [e["step"] for e in events]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(
        steps,
        [e["fraction"] for e in events],
        color=SERIES[0],
        lw=2,
        label="scheduled swap fraction α_t",
    )
    ax.plot(
        steps,
        [1 - e["jaccard"] for e in events],
        "o",
        color=SERIES[1],
        ms=4,
        label="measured churn (1 − Jaccard)",
    )
    ax.set(xlabel="training step", ylabel="fraction of active weights", title=title)
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


PATH_STYLE = {
    "dense": ("dense ×0 (reference, not a speedup)", REFERENCE, "s"),
    "csr": ("CSR (hand-written)", SERIES[0], "o"),
    "compact": ("compacted dense", SERIES[1], "^"),
}


def plot_cost(cost_csv: Path, path: Path) -> None:
    """FLOPs and measured latency vs sparsity for each inference path."""
    rows = read_csv(cost_csv)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [
        ("flops_per_example", "FLOPs per example (log)"),
        ("latency_b1_us", "latency, batch 1, µs (log)"),
        ("latency_b256_us", "latency, batch 256, µs (log)"),
    ]
    for ax, (key, label) in zip(axes, panels, strict=True):
        for name, (legend, color, marker) in PATH_STYLE.items():
            pts = [r for r in rows if r["path"] == name]
            ax.plot(
                [100 * r["target_sparsity"] for r in pts],
                [r[key] for r in pts],
                marker=marker,
                ms=6,
                lw=2,
                color=color,
                label=legend,
            )
        ax.set(xlabel="sparsity (%)", ylabel=label, yscale="log")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "What pruning saves: analytic work vs measured wall-clock (seed-0 Fisher models)", color=INK
    )
    _save(fig, path)


ARM_STYLE = {
    "magnitude": ("magnitude (baseline)", SERIES[0], "o"),
    "fisher": ("Fisher (ours)", SERIES[1], "s"),
    "magnitude+regrow": ("magnitude + Fisher-guided regrowth", SERIES[2], "^"),
}


def _nines(sparsity):
    """x position -log10(1 - s): spreads 90 / 95 / 98 % apart, where the action is."""
    return -np.log10(1.0 - np.asarray(sparsity, dtype=float))


def plot_pareto(
    summary: Sequence[dict], comparisons: Sequence[dict], dense: dict, path: Path
) -> None:
    """Left: accuracy vs sparsity (mean ± std over seeds). Right: paired difference vs
    magnitude with 95 % paired-t CI: the "is it real or noise" view."""
    fig, (ax, ax_d) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    ticks = sorted({r["target_sparsity"] for r in summary})
    levels = [t for t in ticks if t > 0]
    n_seeds = min(r["n_seeds"] for r in summary)
    ax.axhline(100 * dense["mean"], color=REFERENCE, lw=1.5, ls="--")
    ax.text(
        _nines(0.0) + 0.02,
        100 * dense["mean"] + 0.03,
        f"dense {100 * dense['mean']:.2f} %",
        color=INK_MUTED,
        fontsize=8,
    )
    offsets = {"magnitude": -0.012, "fisher": 0.0, "magnitude+regrow": 0.012}
    for arm, (label, color, marker) in ARM_STYLE.items():
        pts = sorted((r for r in summary if r["arm"] == arm), key=lambda r: r["target_sparsity"])
        x = _nines([r["target_sparsity"] for r in pts]) + offsets[arm]
        ax.errorbar(
            x,
            [100 * r["mean"] for r in pts],
            yerr=[100 * r["std"] for r in pts],
            color=color,
            marker=marker,
            ms=6,
            lw=2,
            capsize=3,
            label=label,
        )
    ax.set_xticks(_nines(ticks), [f"{100 * t:g}" for t in ticks])
    ax.set(
        xlabel="sparsity (%), spacing ∝ −log₁₀(1 − s)",
        ylabel="test accuracy (%)",
        title=f"Accuracy vs sparsity (mean ± std, {n_seeds} seeds)",
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax_d.axhline(0.0, color=REFERENCE, lw=1.5)
    for arm in ("fisher", "magnitude+regrow"):
        label, color, marker = ARM_STYLE[arm]
        pts = sorted(
            (c for c in comparisons if c["arm"] == arm and c["baseline"] == "magnitude"),
            key=lambda c: c["target_sparsity"],
        )
        x = _nines([c["target_sparsity"] for c in pts]) + offsets[arm]
        y = np.array([100 * c["mean_diff"] for c in pts])
        lo = y - np.array([100 * c["ci_low"] for c in pts])
        hi = np.array([100 * c["ci_high"] for c in pts]) - y
        ax_d.errorbar(
            x,
            y,
            yerr=[lo, hi],
            color=color,
            marker=marker,
            ms=6,
            lw=0,
            elinewidth=2,
            capsize=4,
            label=f"{label} − magnitude",
        )
    ax_d.set_xticks(_nines(levels), [f"{100 * t:g}" for t in levels])
    ax_d.set(
        xlabel="sparsity (%)",
        ylabel="paired difference (percentage points)",
        title="Difference vs magnitude, 95 % paired-t CI",
    )
    _style(ax_d)
    ax_d.legend(frameon=False, fontsize=8)
    _save(fig, path)
