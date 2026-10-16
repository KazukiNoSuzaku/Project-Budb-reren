"""Part 4: the sparsity sweep across criteria and seeds.

Usage:  uv run python scripts/sweep.py            # full pre-registered sweep (80 runs)
        uv run python scripts/sweep.py --quick    # smoke test: 2 seeds, 3 epochs, 3 levels

Resumable: rows already present in the output CSV are skipped, so an
interrupted sweep continues where it stopped. Every row records its full config.

Writes: results/part4/sweep.csv (the raw numbers; analysed by analyze_sweep.py)
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

from budbaereren.train.data import load_mnist
from budbaereren.train.experiment import RunConfig, run

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0.0, 0.5, 0.75, 0.9, 0.95, 0.98]
SEEDS = [0, 1, 2, 3, 4]
ARMS = {
    "magnitude": {"criterion": "magnitude"},
    "fisher": {"criterion": "fisher"},
    "magnitude+regrow": {"criterion": "magnitude", "regrow_fraction": 0.3},
}
FIELDS = [
    "arm",
    "target_sparsity",
    "seed",
    "test_acc",
    "test_loss",
    "val_acc",
    "val_loss",
    "achieved_sparsity",
    "active_weights",
    "active_params",
    "layer_density",
    "seconds",
    "epochs",
]


def plan(levels, seeds):
    for seed in seeds:
        yield "dense", 0.0, seed, {"criterion": "magnitude"}
    for level in levels:
        if level == 0.0:
            continue
        for arm, changes in ARMS.items():
            for seed in seeds:
                yield arm, level, seed, changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test sweep")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    levels, seeds, epochs = (LEVELS, SEEDS, 20) if not args.quick else ([0.0, 0.5, 0.9], [0, 1], 3)
    out = args.out or ROOT / "results" / "part4" / (
        "sweep_quick.csv" if args.quick else "sweep.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with out.open(newline="") as f:
            done = {
                (r["arm"], float(r["target_sparsity"]), int(r["seed"])) for r in csv.DictReader(f)
            }

    data = load_mnist()
    jobs = [j for j in plan(levels, seeds) if (j[0], j[1], j[2]) not in done]
    print(f"{len(done)} runs already done, {len(jobs)} to go -> {out}", flush=True)
    for i, (arm, level, seed, changes) in enumerate(jobs, 1):
        t0 = time.perf_counter()
        cfg = replace(RunConfig(), target_sparsity=level, seed=seed, epochs=epochs, **changes)
        if args.quick:
            cfg = replace(cfg, prune_begin_epoch=0.5, prune_end_epoch=2.0)
        result, _ = run(cfg, data)
        row = {k: result[k] for k in FIELDS if k in result}
        row |= {
            "arm": arm,
            "target_sparsity": level,
            "seed": seed,
            "epochs": epochs,
            "layer_density": " ".join(f"{d:.5f}" for d in result["layer_density"]),
        }
        new_file = not out.exists()
        with out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            if new_file:
                writer.writeheader()
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()}
            )
        print(
            f"[{i}/{len(jobs)}] {arm:17s} {level:4.0%} seed {seed}: test acc "
            f"{result['test_acc']:.4f} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
