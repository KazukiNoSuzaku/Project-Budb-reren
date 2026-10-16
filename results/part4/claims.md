# Pre-registered hypotheses, evaluated mechanically

Source: `results/part4/sweep.csv`. Decision rule: a difference is *real* if the 95 % paired-t CI over seeds excludes 0.

| hypothesis | mean diff (pp) | 95 % paired-t CI (pp) | bootstrap CI (pp) | wins | verdict |
|---|---:|---:|---:|---:|---|
| H1 (primary): Fisher > magnitude at 98 % | -0.008 | [-0.199, +0.183] | [-0.122, +0.122] | 2/5 | not distinguishable from seed noise |
| H2: magnitude+regrow > magnitude at 98 % | +0.098 | [-0.083, +0.279] | [-0.014, +0.214] | 4/5 | not distinguishable from seed noise |
| H3: magnitude ≥ Fisher at 95 % | +0.026 | [-0.167, +0.219] | [-0.092, +0.144] | 3/5 | not distinguishable from seed noise |

Dense baseline: 98.40 % ± 0.09 (std over 5 seeds).
