# Budbæreren — Design

This document answers the four design questions from the brief, with the
reasoning behind each choice. The build history and smaller decisions are in
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

1. [The gradient of a masked weight](#1-the-gradient-of-a-masked-weight)
2. [The importance criterion](#2-the-importance-criterion)
3. [Where the engine bottlenecks](#3-where-the-engine-bottlenecks) *(Phase 9)*
4. [Serving a self-pruned model at scale](#4-serving-a-self-pruned-model-at-scale) *(Phase 9)*

---

## 1. The gradient of a masked weight

### What the engine computes

Every prunable layer computes with the **effective weight**
`W_eff = W ⊙ M`, where `M ∈ {0,1}` is the mask stored on the parameter. The
autodiff engine differentiates that expression like any other, so by the
chain rule

```
∂L/∂W  = ∂L/∂W_eff ⊙ M          (parameter gradient)
∂L/∂W_eff = (layer input)ᵀ · (upstream gradient)   (dense gradient)
```

- **For the optimizer, the gradient of a masked weight is exactly `0`.**
  Not a small number: the engine multiplies by an exact `0.0`. This is the
  mathematically correct derivative. A pruned `W_ij` is multiplied by zero in
  the forward pass, so the loss does not depend on it at all.
- **For pruning decisions, we also keep the dense gradient `∂L/∂W_eff`**,
  which is generally non-zero at masked positions. It answers a different
  question: *if this connection existed (at its current value, 0), which way
  would the loss want it to move, and how strongly?* That is exactly the
  signal regrowth needs (RigL-style). `Linear` marks `W_eff` with
  `retain_grad()` so both are available after one backward pass.

### Why this is the right choice

There are three plausible alternatives; each is wrong for a specific reason.

| Alternative | What goes wrong |
|---|---|
| Treat the masked weight's gradient as the dense gradient (straight-through) | The optimizer would update a weight that is not in the network. Its value then becomes a hidden variable that "remembers" training it did not participate in. |
| Undefined / `None` for masked entries | Breaks vectorised optimizers and makes the zero ambiguous ("never computed" vs "computed as 0"). |
| Zero gradient, but a mask-unaware optimizer | **The common bug.** See below. |

**A zero gradient does not freeze a weight under momentum.** Adam's update
is `lr · m̂ / (√v̂ + ε)`, and `m` still contains momentum accumulated before
the prune. With `g = 0` the weight keeps moving: in our evidence run, a
"pruned" weight drifts to `|w| = 0.069` within 50 steps while receiving an
exactly-zero gradient every step (`results/masking_evidence.csv`). The
sparsity is then fake: the stored weight is non-zero, and only the forward
mask hides it.

So the gradient semantics are only half of the story. The other half is the
**optimizer contract** (`budbaereren/optim/optimizer.py`):

1. Masked entries are skipped entirely: no parameter step, no `m`/`v`
   update (not even decay), no weight decay, no step-count increment. After
   each step the optimizer re-asserts `W[~M] = 0.0`.
2. A pruned entry's optimizer state is frozen and never read.
3. On revival (`M: 0 → 1`, detected by the optimizer itself by diffing
   masks), the entry's `m`, `v` **and its own step counter `t`** are reset,
   and the weight restarts from `0.0`.

### Why Adam needs a per-element step counter

Bias correction divides by `1 − βᵗ`. Suppose we reset `m = v = 0` on
revival but keep one global `t`. On the first step after revival with
gradient `g`:

```
m̂ = (1−β₁) g / (1−β₁ᵗ),   v̂ = (1−β₂) g² / (1−β₂ᵗ)

|Δw| / lr = [(1−β₁)/(1−β₁ᵗ)] / √[(1−β₂)/(1−β₂ᵗ)]
          → (1−β₁)/√(1−β₂) = 0.1/√0.001 ≈ 3.16   for t ≫ 1/(1−β₂) = 1000
```

and ≈ 0.48 at `t = 10`, ≈ 0.98 at `t = 100`. The revived weight's first
step depends on *how long training has been running*, which has nothing to
do with that weight. With a per-element counter restarted at `t = 1`,
`m̂ = g`, `v̂ = g²`, and `|Δw| = lr` exactly, the same as a freshly created
parameter.

If instead the state is *not* reset at all, the momentum from before pruning
dominates: in the evidence run the gradient after revival points one way and
the stale `m` points the other, and the first step goes the **wrong way**
(`−0.8 lr`).

| Variant (`results/masking_evidence.csv`) | drift while pruned | first step after revival |
|---|---:|---:|
| **Budbæreren Adam** (mask-aware, per-element `t`) | **0** | **+1.000 lr** |
| naive: mask the gradient only | 0.069 | n/a (never really pruned) |
| naive: no state reset on revival | 0 | −0.800 lr |
| naive: reset `m, v`, global `t` | 0 | +3.152 lr |

All four behaviours are asserted in `tests/test_masking.py`.

---

## 2. The importance criterion

### What "importance" should mean

The question a pruner must answer for each connection `i` is: **how much
does the loss go up if I remove it?** Removing `wᵢ` sets it to zero, i.e.
applies the perturbation `δw = −wᵢ eᵢ`. Expand the loss to second order
around the current weights:

```
L(w + δw) = L(w) + gᵀδw + ½ δwᵀ H δw + O(‖δw‖³)

ΔLᵢ ≈ −gᵢ wᵢ + ½ Hᵢᵢ wᵢ²            (δw has a single non-zero entry)
```

`g = ∇L` and `H = ∇²L`. With only one non-zero entry in `δw`, only the
diagonal `Hᵢᵢ` survives. Removing *many* weights at once would also bring in
the cross terms `Hᵢⱼ wᵢ wⱼ`, which this (Optimal Brain Damage) approximation
ignores. That is one reason to prune **gradually**: each event removes a
small set, and the network re-adapts before the next one.

Each criterion is an estimate of this `ΔLᵢ`:

| criterion | score | what it assumes |
|---|---|---|
| `magnitude` | `|wᵢ|` | `g = 0` and the same `Hᵢᵢ` for every weight, so `ΔL ∝ wᵢ²` |
| `taylor1` | `|wᵢ · E[gᵢ]|` | only the first-order term matters |
| `fisher` | `½ wᵢ² · E[gᵢ²]` | only the second-order term matters, with `Hᵢᵢ ≈ E[gᵢ²]` |
| `taylor2` | `|−wᵢ E[gᵢ] + ½ wᵢ² E[gᵢ²]|` | both terms |

### Why each approximation is reasonable, and where it breaks

**The first-order term `−gᵢwᵢ`.** During training `E[g] ≠ 0`, so this term
is informative. But training *drives* `E[g] → 0`. At a converged minimum the
first-order term vanishes for every weight and stops discriminating, which
is exactly why Optimal Brain Damage (LeCun et al., 1990), which prunes a
trained network, needed the second-order term. A single minibatch gradient
is mostly sampling noise around `E[g]`, so we estimate `E[g]` with an EMA
(β = 0.99, ~100 steps).

**The second-order term `½ Hᵢᵢ wᵢ²`.** Computing `Hᵢᵢ` exactly would need a
second backward pass per weight. For a negative log-likelihood loss, the
expected Hessian equals the Fisher information (the Gauss-Newton matrix),
`E[H] = E[∇ℓ ∇ℓᵀ]`, whose diagonal is the expected *squared* gradient
`E[gᵢ²]`. That is first-order cost: we already have `g` every step. We use
an EMA of squared minibatch gradients, which is exactly what Adam tracks as
`v`. (This estimates `E[g]² + Var[g]/B` rather than the per-sample Fisher
`E[ℓᵢ'²]`. The two differ by a batch-size factor on the variance part, but
the *ranking* of weights, which is all a pruner needs, is preserved to the
extent the per-weight noise structure is.) Unlike the first-order term, this
one does **not** vanish at a minimum: a weight with high gradient variance
is one the loss is sensitive to.

**Why magnitude should be the weaker baseline.** For a weight `wᵢⱼ` from
input unit `i` to unit `j`, the minibatch gradient is `gᵢⱼ = mean_batch(xᵢ δⱼ)`,
where `δⱼ` is the backpropagated error at `j`. So `E[gᵢⱼ²]` grows with
**how active its input is** and **how sensitive the loss is to its output**,
and the Fisher score multiplies the weight's size by exactly that. Magnitude keeps only
the first factor. A large weight from an input that is rarely active, or
into a unit whose output the loss barely depends on, is cheap to remove.
Magnitude cannot see that; a gradient-based score can. On MNIST this is not
a corner case: 289 of the 784 pixels (37 %) are non-zero in fewer than 1 %
of training images.

### Why EMAs rather than one batch

A score computed from a single minibatch is a noisy sample of the expected
score. Averaging over ~100 steps (β = 0.99) reduces the standard deviation of
that noise by roughly √100 = 10× (for weakly correlated batches) while still
tracking the weights as they move. Whether this matters in practice is
measured, not assumed: Phase 6 ablates β = 0 (single-batch scores).
The prune frequency (every 100 steps) matches the EMA horizon, so each
decision uses statistics gathered since the previous event.

### How the choice was made

Before running anything on the test set, all criteria were compared on
the **validation** split with seeds (100, 101) that are disjoint from the
reporting seeds (0–4); `scripts/select_criterion.py`,
`results/part3/criterion_selection.csv`. Mean validation accuracy:

| target sparsity | magnitude | taylor1 | **fisher** | taylor2 | random |
|---:|---:|---:|---:|---:|---:|
| 95 % | **0.9836** | 0.9817 | 0.9815 | 0.9816 | 0.9508 |
| 98 % | 0.9790 | 0.9800 | **0.9806** | 0.9796 | 0.6801 |

The pre-declared rule was "the gradient-based criterion with the best mean
validation accuracy", which picks **Fisher** (`½ w² E[g²]`).

Two honest observations:

1. **Magnitude is not beaten everywhere.** At 95 % it is the best criterion
   on validation; the gradient criteria only pull ahead at 98 %, where the
   budget is tight enough that *which* weights survive matters more. With
   two seeds none of these gaps is statistically established. That is the
   job of the Phase 8 multi-seed test-set comparison.
2. **Every criterion is far above random** (0.68 vs ≈ 0.98 at 98 %), so the
   pipeline is measuring something real, and the differences between real
   criteria are small compared with the gap to an uninformed choice.

**The test-set verdict (Phase 8, five fresh seeds, pre-registered rule): at
98 % sparsity Fisher and magnitude are statistically tied**, −0.01 points,
95 % paired-t CI [−0.20, +0.18] (`results/part4/claims.md`). The validation
edge for Fisher was seed noise. The derivation above explains why a
loss-aware score *should* help. On this task, with gradual pruning and a
shared global allocation, the measured benefit is below what five seeds
can resolve (~0.2 points). We report that rather than the validation-set win.

A plausible reason gradual magnitude pruning is a strong baseline here: with
a cubic schedule and Adam, weights that matter grow under training pressure
and weights that do not drift little, so after a few epochs magnitude is
already correlated with the loss-based score. The cases where the two differ
most (low-activity inputs, insensitive units) matter most when very few
weights may remain.

### Regrowth and what the ablations add

Pruned connections can come back (RigL-style, `budbaereren/prune/regrowth.py`).
The grow signal is the EMA of the squared **dense** gradient
`E[(∂L/∂W_eff)²]` of a dead connection: the same Fisher quantity as the
criterion, evaluated where `w = 0`, so it measures how much the loss
*wants* the connection rather than how much it would lose. Dropping uses the
criterion score. The swap fraction decays with a cosine, because otherwise
churn stays at `2α/(1+α)` forever and the network never settles. See
IMPLEMENTATION.md, Phase 6, for the stability analysis and a same-step
prune/regrow bug that would have reintroduced stale optimizer state.

The validation ablations put the criterion choice in perspective: at 98 %
sparsity, the schedule (cubic vs one-shot: 4.6 points), score smoothing
(EMA vs single batch: 0.9) and allocation (global vs layer-wise: 0.4–0.8)
each matter **more** than magnitude vs Fisher (0.2). The best validation
variant was magnitude dropping with Fisher-guided growth, which suggests the
two signals are complementary: magnitude is a stable *drop* rule, gradient
information is a good *grow* rule.

## 3. Where the engine bottlenecks

*Written in Phase 9.*

## 4. Serving a self-pruned model at scale

*Written in Phase 9.*
