"""Phase 6 ablations on the VALIDATION set with held-out seeds (100, 101).

Each variant changes one design choice relative to the default
(Fisher, global allocation, cubic schedule, EMA beta 0.99, no regrowth).

Usage:  uv run python scripts/ablations.py
Writes: results/part3/ablations.csv  (one row per run)
        results/part3/ablations.md   (mean validation accuracy per variant)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from budbaereren.train.data import load_mnist
from budbaereren.train.experiment import RunConfig, run
from budbaereren.train.metrics import write_csv

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [100, 101]
TARGETS = [0.95, 0.98]
BASE = RunConfig(criterion="fisher")
VARIANTS = {
    "fisher (default)": {},
    "fisher + regrowth 0.3": {"regrow_fraction": 0.3},
    "fisher, single-batch scores (beta=0)": {"ema_beta": 0.0},
    "fisher, layer-wise allocation": {"scope": "layerwise"},
    "fisher, one-shot at end of ramp": {"schedule": "oneshot"},
    "magnitude": {"criterion": "magnitude"},
    "magnitude, layer-wise allocation": {"criterion": "magnitude", "scope": "layerwise"},
    "magnitude + regrowth 0.3": {"criterion": "magnitude", "regrow_fraction": 0.3},
}


def main() -> None:
    data = load_mnist()
    rows = []
    for target in TARGETS:
        for name, changes in VARIANTS.items():
            for seed in SEEDS:
                cfg = replace(BASE, target_sparsity=target, seed=seed, **changes)
                result, _ = run(cfg, data)
                churn = [1 - e["jaccard"] for e in result["regrowth_events"]]
                rows.append(
                    {
                        "variant": name,
                        "target_sparsity": target,
                        "seed": seed,
                        "val_acc": result["val_acc"],
                        "val_loss": result["val_loss"],
                        "layer_density": " ".join(f"{d:.4f}" for d in result["layer_density"]),
                        "regrown_total": sum(e["regrown"] for e in result["regrowth_events"]),
                        "final_churn": churn[-1] if churn else 0.0,
                    }
                )
                print(
                    f"{target:.2f} {name:40s} seed {seed}: val acc {result['val_acc']:.4f}",
                    flush=True,
                )
    write_csv(rows, ROOT / "results" / "part3" / "ablations.csv")

    lines = [
        "| variant | " + " | ".join(f"val acc @ {t:.0%}" for t in TARGETS) + " |",
        "|---|" + "---:|" * len(TARGETS),
    ]
    for name in VARIANTS:
        cells = []
        for target in TARGETS:
            accs = [
                r["val_acc"]
                for r in rows
                if r["variant"] == name and r["target_sparsity"] == target
            ]
            cells.append(f"{np.mean(accs):.4f} ± {np.std(accs, ddof=1):.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines) + f"\n\nMean ± std over seeds {SEEDS}, validation split.\n"
    (ROOT / "results" / "part3" / "ablations.md").write_text(table)
    print("\n" + table)


if __name__ == "__main__":
    main()
