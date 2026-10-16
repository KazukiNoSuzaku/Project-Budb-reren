"""Part 2: train the dense LeNet-300-100 on MNIST with Adam.

Usage:  uv run python scripts/train_dense.py [--epochs 20] [--seed 0]

Writes to results/part2/:
  history.csv          per-epoch train/val loss and accuracy
  learning_curve.png   loss and accuracy curves
  summary.json         config, test accuracy, initial loss, wall time
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from budbaereren.nn import MLP
from budbaereren.optim import Adam
from budbaereren.train.data import load_mnist
from budbaereren.train.loop import cosine_lr, evaluate, train
from budbaereren.train.metrics import plot_learning_curve, write_csv, write_json
from budbaereren.utils import seed_everything

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, nargs="+", default=[300, 100])
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "part2")
    args = parser.parse_args()

    rng = seed_everything(args.seed)
    data = load_mnist()
    model = MLP([784, *args.hidden, 10], rng)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(model, f"| {model.num_parameters():,} parameters")

    steps = args.epochs * -(-len(data.train) // args.batch_size)
    history = train(
        model,
        optimizer,
        data,
        args.epochs,
        args.batch_size,
        rng,
        lr_schedule=cosine_lr(args.lr, steps),
    )
    test_loss, test_acc = evaluate(model, data.test)
    print(f"test loss {test_loss:.4f} | test accuracy {test_acc:.4f}")

    write_csv(history, args.out / "history.csv")
    plot_learning_curve(
        history,
        args.out / "learning_curve.png",
        f"Dense {model!r}, Adam lr={args.lr} (cosine), seed {args.seed}",
    )
    write_json(
        {
            "config": {k: v for k, v in vars(args).items() if k != "out"}
            | {"lr_schedule": "cosine"},
            "parameters": model.num_parameters(),
            "initial_val_loss": history[0]["val_loss"],
            "chance_level_loss": float(np.log(10)),
            "best_val_acc": max(r["val_acc"] for r in history),
            "final_val_acc": history[-1]["val_acc"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "train_seconds": history[-1]["seconds"],
            "nan_free": True,  # train() raises FloatingPointError otherwise
        },
        args.out / "summary.json",
    )


if __name__ == "__main__":
    main()
