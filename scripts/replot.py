"""Regenerate every committed figure from the committed CSVs (no training).

Usage:  uv run python scripts/replot.py
"""

from __future__ import annotations

import json
from pathlib import Path

from budbaereren.train.metrics import (
    plot_cost,
    plot_learning_curve,
    plot_pruning_run,
    plot_regrowth,
    read_csv,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    part2 = RESULTS / "part2"
    if (part2 / "history.csv").exists():
        cfg = json.loads((part2 / "summary.json").read_text())["config"]
        plot_learning_curve(
            read_csv(part2 / "history.csv"),
            part2 / "learning_curve.png",
            f"Dense MLP(784-300-100-10), Adam lr={cfg['lr']} (cosine), seed {cfg['seed']}",
        )
    for run_dir in sorted(d for d in (RESULTS / "part3").glob("*_s*") if d.is_dir()):
        cfg = json.loads((run_dir / "summary.json").read_text())["config"]
        title = (
            f"{cfg['criterion']} → {cfg['target_sparsity']:.0%} sparsity, seed {cfg['seed']}"
            + (", with regrowth" if cfg.get("regrow_fraction") else "")
        )
        plot_pruning_run(
            run_dir / "history.csv", run_dir / "events.csv", run_dir / "pruning.png", title
        )
        if (run_dir / "regrowth.csv").exists():
            plot_regrowth(
                run_dir / "regrowth.csv",
                run_dir / "regrowth.png",
                f"regrowth churn, {cfg['criterion']} → {cfg['target_sparsity']:.0%}",
            )
    if (RESULTS / "part4" / "cost.csv").exists():
        plot_cost(RESULTS / "part4" / "cost.csv", RESULTS / "part4" / "cost.png")
    print("figures regenerated")


if __name__ == "__main__":
    main()
