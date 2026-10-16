"""Choose the importance criterion on the VALIDATION set, before any test-set sweep.

Seeds used here (100, 101) are disjoint from the seeds used for reported
results (0-4), and only validation accuracy is used to choose. The test-set
numbers that Part 4 reports therefore cannot have influenced the choice.

Usage:  uv run python scripts/select_criterion.py
Writes: results/part3/criterion_selection.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from budbaereren.train.data import load_mnist
from budbaereren.train.experiment import RunConfig, run
from budbaereren.train.metrics import write_csv

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ["magnitude", "taylor1", "fisher", "taylor2", "random"]
TARGETS = [0.95, 0.98]
SEEDS = [100, 101]


def main() -> None:
    data = load_mnist()
    rows = []
    for target in TARGETS:
        for criterion in CRITERIA:
            for seed in SEEDS:
                result, _ = run(
                    RunConfig(criterion=criterion, target_sparsity=target, seed=seed), data
                )
                rows.append(
                    {
                        "criterion": criterion,
                        "target_sparsity": target,
                        "seed": seed,
                        "val_acc": result["val_acc"],
                        "val_loss": result["val_loss"],
                    }
                )
                print(
                    f"{target:.2f} {criterion:10s} seed {seed}: val acc {result['val_acc']:.4f}",
                    flush=True,
                )
    write_csv(rows, ROOT / "results" / "part3" / "criterion_selection.csv")
    print("\nmean validation accuracy")
    for target in TARGETS:
        for criterion in CRITERIA:
            accs = [
                r["val_acc"]
                for r in rows
                if r["criterion"] == criterion and r["target_sparsity"] == target
            ]
            print(f"  {target:.2f} {criterion:10s} {np.mean(accs):.4f}")


if __name__ == "__main__":
    main()
