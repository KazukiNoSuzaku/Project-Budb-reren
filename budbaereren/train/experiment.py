"""One self-pruning training run, fully described by a config.

Both the Part 3 script and the Part 4 sweep go through ``run``, so every
number in the repository comes from the same code path.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from budbaereren.nn import MLP
from budbaereren.optim import Adam
from budbaereren.prune import CubicSchedule, OneShotSchedule, Pruner, make_criterion
from budbaereren.train.data import MNIST
from budbaereren.train.loop import cosine_lr, evaluate, train
from budbaereren.utils import seed_everything


@dataclass(frozen=True)
class RunConfig:
    criterion: str = "fisher"
    target_sparsity: float = 0.9
    seed: int = 0
    hidden: tuple[int, ...] = (300, 100)
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    schedule: str = "cubic"  # "cubic" | "oneshot"
    prune_begin_epoch: float = 1.0
    prune_end_epoch: float = 12.0
    prune_frequency: int = 100
    scope: str = "global"  # "global" | "layerwise"
    min_density: float = 0.01
    ema_beta: float = 0.99

    def steps_per_epoch(self, n_train: int) -> int:
        return -(-n_train // self.batch_size)


def build_schedule(cfg: RunConfig, steps_per_epoch: int):
    begin = int(round(cfg.prune_begin_epoch * steps_per_epoch))
    end = int(round(cfg.prune_end_epoch * steps_per_epoch))
    if cfg.schedule == "cubic":
        return CubicSchedule(cfg.target_sparsity, begin, end, cfg.prune_frequency)
    if cfg.schedule == "oneshot":
        return OneShotSchedule(cfg.target_sparsity, end)
    raise ValueError(f"unknown schedule {cfg.schedule!r}")


def run(cfg: RunConfig, data: MNIST, log=None) -> tuple[dict, MLP]:
    """Train with self-pruning; return (result dict, trained model)."""
    rng = seed_everything(cfg.seed)
    model = MLP([data.train.x.shape[1], *cfg.hidden, 10], rng)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    spe = cfg.steps_per_epoch(len(data.train))
    total_steps = cfg.epochs * spe

    pruner = None
    if cfg.target_sparsity > 0:
        kwargs = {"beta": cfg.ema_beta}
        if cfg.criterion == "random":
            kwargs["seed"] = cfg.seed
        pruner = Pruner(
            model.prunable_parameters(),
            make_criterion(cfg.criterion, **kwargs),
            build_schedule(cfg, spe),
            scope=cfg.scope,
            min_density=cfg.min_density,
        )

    def epoch_metrics(epoch: int) -> dict:
        if pruner is None:
            return {"sparsity": 0.0}
        return {"sparsity": pruner.sparsity()}

    t0 = time.perf_counter()
    history = train(
        model,
        optimizer,
        data,
        cfg.epochs,
        cfg.batch_size,
        rng,
        step_callback=pruner,
        epoch_callback=epoch_metrics,
        lr_schedule=cosine_lr(cfg.lr, total_steps),
        log=log,
    )
    seconds = time.perf_counter() - t0

    val_loss, val_acc = evaluate(model, data.val)
    test_loss, test_acc = evaluate(model, data.test)
    weights = model.prunable_parameters()
    n_weights = sum(p.size for p in weights)
    active_weights = sum(int(np.count_nonzero(p.data)) for p in weights)
    n_bias = sum(p.size for p in model.parameters() if not p.prunable)
    result = {
        "config": asdict(cfg),
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "achieved_sparsity": 1.0 - active_weights / n_weights,
        "active_weights": active_weights,
        "total_weights": n_weights,
        "active_params": active_weights + n_bias,
        "layer_density": [float(np.count_nonzero(p.data)) / p.size for p in weights],
        "seconds": seconds,
        "history": history,
        "events": [] if pruner is None else [asdict(e) for e in pruner.events],
    }
    return result, model
