"""Part 3: train LeNet-300-100 on MNIST while it prunes itself to a target sparsity.

Usage:  uv run python scripts/train_prune.py [--target 0.9] [--criterion fisher] [--seed 0]

Writes to results/part3/<criterion>_s<target>/ :
  history.csv       per-epoch loss / accuracy / sparsity
  events.csv        every prune event: step, target, achieved sparsity, per-layer density
  pruning.png       sparsity schedule and validation accuracy over training
  summary.json      config, final test accuracy, achieved sparsity, active parameters
  model.npz         final weights and masks (used by the Phase 7 cost benchmark)
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from budbaereren.prune import CRITERIA
from budbaereren.train.data import load_mnist
from budbaereren.train.experiment import RunConfig, run
from budbaereren.train.metrics import plot_pruning_run, write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=float, default=0.9, help="final sparsity in [0, 1)")
    parser.add_argument("--criterion", choices=sorted(CRITERIA), default="fisher")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--schedule", choices=["cubic", "oneshot"], default="cubic")
    parser.add_argument("--scope", choices=["global", "layerwise"], default="global")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = RunConfig(
        criterion=args.criterion,
        target_sparsity=args.target,
        seed=args.seed,
        epochs=args.epochs,
        schedule=args.schedule,
        scope=args.scope,
    )
    out = args.out or ROOT / "results" / "part3" / f"{args.criterion}_s{round(args.target * 100)}"
    print(" | ".join(f"{f.name}={getattr(cfg, f.name)}" for f in fields(cfg)))

    result, model = run(cfg, load_mnist(), log=print)
    print(
        f"test accuracy {result['test_acc']:.4f} | achieved sparsity "
        f"{result['achieved_sparsity']:.4f} | active weights {result['active_weights']:,}"
        f" / {result['total_weights']:,} | layer density "
        + ", ".join(f"{d:.3f}" for d in result["layer_density"])
    )

    write_csv(result.pop("history"), out / "history.csv")
    events = result.pop("events")
    write_csv(
        [
            {
                **{k: v for k, v in e.items() if k != "layer_density"},
                **{f"density_l{i}": d for i, d in enumerate(e["layer_density"])},
            }
            for e in events
        ],
        out / "events.csv",
    )
    write_json(result, out / "summary.json")
    plot_pruning_run(
        out / "history.csv",
        out / "events.csv",
        out / "pruning.png",
        f"{args.criterion} → {args.target:.0%} sparsity, seed {args.seed}",
    )
    np.savez_compressed(out / "model.npz", **model.state_dict())
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
