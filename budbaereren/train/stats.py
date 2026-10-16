"""Paired statistics across seeds, in plain NumPy (no SciPy).

For a fixed seed all arms share the initial weights and the minibatch order,
so per-seed differences d_i = a_i - b_i cancel the seed effect. We report:

* the paired-t 95 % confidence interval  d̄ ± t_{0.975, n-1} · s_d / √n
  (the pre-registered decision rule: "real" iff it excludes 0);
* a percentile bootstrap 95 % CI of d̄ (secondary; crude for n = 5);
* the sign count (in how many seeds A beat B).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Two-sided 95 % critical values of Student's t, t_{0.975, df}, df = 1..30.
_T975 = [
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
]


def t_critical_975(df: int) -> float:
    if df < 1:
        raise ValueError("need at least 2 paired samples")
    return _T975[df - 1] if df <= len(_T975) else 1.960


@dataclass
class PairedComparison:
    n: int
    mean_diff: float
    sd_diff: float
    t_stat: float
    ci_low: float
    ci_high: float
    boot_low: float
    boot_high: float
    wins: int  # seeds where a > b
    real: bool  # paired-t CI excludes 0

    def as_dict(self) -> dict:
        return asdict(self)


def paired_comparison(a, b, n_boot: int = 10_000, seed: int = 0) -> PairedComparison:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("need two 1-D arrays of paired observations")
    d = a - b
    n = d.size
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    half = t_critical_975(n - 1) * se
    if se > 0:
        t_stat = mean / se
    else:  # identical differences in every seed
        t_stat = float(np.copysign(np.inf, mean)) if mean else 0.0
    boot = np.random.default_rng(seed).choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return PairedComparison(
        n=n,
        mean_diff=mean,
        sd_diff=sd,
        t_stat=float(t_stat),
        ci_low=mean - half,
        ci_high=mean + half,
        boot_low=float(lo),
        boot_high=float(hi),
        wins=int((d > 0).sum()),
        real=bool(mean - half > 0 or mean + half < 0),
    )
