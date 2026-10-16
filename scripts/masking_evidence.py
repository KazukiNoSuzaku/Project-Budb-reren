"""Quantify what goes wrong with naive masked-optimizer implementations.

Writes results/masking_evidence.csv. Each row is one optimizer variant run on
the same controlled scenario (see tests/test_masking.py for the assertions):

* drift_after_50_steps: max |w| of pruned weights after 50 steps while pruned
  (should be exactly 0).
* first_step_after_revival_in_lr: the first post-revival weight change in units
  of lr. A freshly initialised Adam parameter moves exactly +1.0 lr here
  (new gradient is -1, so the weight should increase).

Usage:  uv run python scripts/masking_evidence.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from test_masking import (  # noqa: E402
    LR,
    GlobalStepAdam,
    MaskUnawareAdam,
    StaleStateAdam,
    isolated_param_run,
    make_problem,
    prune_entries,
    train_steps,
)

from budbaereren.optim import Adam  # noqa: E402

VARIANTS = {
    "budbaereren Adam (mask-aware, per-element t)": Adam,
    "naive: mask gradient only": MaskUnawareAdam,
    "naive: no state reset on revival": StaleStateAdam,
    "naive: reset m,v but global t": GlobalStepAdam,
}


def drift(opt_cls) -> float:
    model, x, y = make_problem()
    opt = opt_cls(model.parameters(), lr=LR)
    train_steps(model, opt, x, y, 20)
    w = model.prunable_parameters()[0]
    pruned = prune_entries(w, [(0, 0), (1, 3), (5, 7), (3, 2)])
    train_steps(model, opt, x, y, 50)
    return float(np.max(np.abs(w.data[pruned])))


def revival_step(opt_cls) -> float:
    if opt_cls is MaskUnawareAdam:
        return float("nan")  # never actually removes the weight, so "revival" is meaningless
    delta, _ = isolated_param_run(opt_cls)
    return float(delta / LR)


def main() -> None:
    out = ROOT / "results" / "masking_evidence.csv"
    out.parent.mkdir(exist_ok=True)
    rows = [(name, drift(cls), revival_step(cls)) for name, cls in VARIANTS.items()]
    with out.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["variant", "drift_after_50_steps", "first_step_after_revival_in_lr"])
        for name, d, r in rows:
            writer.writerow([name, f"{d:.6g}", f"{r:.4f}"])
    width = max(len(n) for n in VARIANTS)
    print(f"{'variant':<{width}}  {'drift':>10}  {'revival step / lr':>18}")
    for name, d, r in rows:
        print(f"{name:<{width}}  {d:>10.3g}  {r:>18.4f}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
