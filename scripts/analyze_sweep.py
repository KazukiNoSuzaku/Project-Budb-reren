"""Analyse the Part 4 sweep: Pareto summary, paired comparisons, plot, claims.

Usage:  uv run python scripts/analyze_sweep.py [--sweep results/part4/sweep.csv]

Writes next to the sweep CSV:
  pareto.csv       per arm and sparsity: mean / std / min / max test accuracy over seeds
  comparison.csv   paired comparisons (by seed) with paired-t and bootstrap 95 % CIs
  pareto.png       accuracy vs sparsity, and paired differences vs magnitude
  claims.md        the pre-registered hypotheses evaluated mechanically
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from budbaereren.train.metrics import plot_pareto, read_csv, write_csv
from budbaereren.train.stats import paired_comparison

ROOT = Path(__file__).resolve().parents[1]
ARMS = ["magnitude", "fisher", "magnitude+regrow"]
PAIRS = [("fisher", "magnitude"), ("magnitude+regrow", "magnitude"), ("fisher", "magnitude+regrow")]
HYPOTHESES = [
    ("H1 (primary)", 0.98, "fisher", "magnitude", "Fisher > magnitude at 98 %"),
    ("H2", 0.98, "magnitude+regrow", "magnitude", "magnitude+regrow > magnitude at 98 %"),
    ("H3", 0.95, "magnitude", "fisher", "magnitude ≥ Fisher at 95 %"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", type=Path, default=ROOT / "results" / "part4" / "sweep.csv")
    args = parser.parse_args()
    out = args.sweep.parent
    rows = read_csv(args.sweep)

    acc: dict[tuple[str, float], dict[int, float]] = defaultdict(dict)
    sparsity: dict[tuple[str, float], list[float]] = defaultdict(list)
    params: dict[tuple[str, float], list[int]] = defaultdict(list)
    for r in rows:
        key = (r["arm"], float(r["target_sparsity"]))
        acc[key][int(r["seed"])] = float(r["test_acc"])
        sparsity[key].append(float(r["achieved_sparsity"]))
        params[key].append(int(r["active_params"]))

    dense_accs = np.array(list(acc[("dense", 0.0)].values()))
    dense = {"mean": float(dense_accs.mean()), "std": float(dense_accs.std(ddof=1))}
    levels = sorted({lvl for arm, lvl in acc if arm != "dense"})

    summary = []
    for arm in ARMS:
        for level in [0.0, *levels]:
            key = ("dense", 0.0) if level == 0.0 else (arm, level)
            a = np.array(list(acc[key].values()))
            summary.append(
                {
                    "arm": arm,
                    "target_sparsity": level,
                    "achieved_sparsity": float(np.mean(sparsity[key])),
                    "active_params": int(np.mean(params[key])),
                    "n_seeds": a.size,
                    "mean": float(a.mean()),
                    "std": float(a.std(ddof=1)),
                    "min": float(a.min()),
                    "max": float(a.max()),
                    "drop_vs_dense": dense["mean"] - float(a.mean()),
                }
            )
    write_csv(summary, out / "pareto.csv")

    comparisons = []
    for level in levels:
        for arm, base in PAIRS:
            seeds = sorted(set(acc[(arm, level)]) & set(acc[(base, level)]))
            if len(seeds) < 2:
                continue
            c = paired_comparison(
                [acc[(arm, level)][s] for s in seeds], [acc[(base, level)][s] for s in seeds]
            )
            comparisons.append(
                {"target_sparsity": level, "arm": arm, "baseline": base, **c.as_dict()}
            )
    write_csv(comparisons, out / "comparison.csv")
    plot_pareto(summary, comparisons, dense, out / "pareto.png")

    lines = [
        "# Pre-registered hypotheses, evaluated mechanically",
        "",
        f"Source: `{args.sweep.resolve().relative_to(ROOT)}`. "
        "Decision rule: a difference is *real* "
        "if the 95 % paired-t CI over seeds excludes 0.",
        "",
        "| hypothesis | mean diff (pp) | 95 % paired-t CI (pp) | bootstrap CI (pp) | "
        "wins | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, level, arm, base, text in HYPOTHESES:
        match = [
            c
            for c in comparisons
            if c["target_sparsity"] == level and {c["arm"], c["baseline"]} == {arm, base}
        ]
        if not match:
            continue
        c = match[0]
        sign = 1 if c["arm"] == arm else -1  # orient as "arm − base"
        d, lo, hi = (100 * sign * c[k] for k in ("mean_diff", "ci_low", "ci_high"))
        blo, bhi = (100 * sign * c[k] for k in ("boot_low", "boot_high"))
        lo, hi, blo, bhi = min(lo, hi), max(lo, hi), min(blo, bhi), max(blo, bhi)
        wins = c["wins"] if sign == 1 else c["n"] - c["wins"]
        if lo > 0:
            verdict = "supported (CI excludes 0)"
        elif hi < 0:
            verdict = "contradicted (CI excludes 0, opposite sign)"
        else:
            verdict = "not distinguishable from seed noise"
        lines.append(
            f"| {name}: {text} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
            f"[{blo:+.3f}, {bhi:+.3f}] | {wins}/{c['n']} | {verdict} |"
        )
    lines += [
        "",
        f"Dense baseline: {100 * dense['mean']:.2f} % ± {100 * dense['std']:.2f} "
        f"(std over {dense_accs.size} seeds).",
    ]
    (out / "claims.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
