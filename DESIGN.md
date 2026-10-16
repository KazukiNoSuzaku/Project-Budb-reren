# Budbæreren — Design

This document answers the four design questions from the brief, with the
reasoning behind each choice. The build history and smaller decisions are in
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

1. [The gradient of a masked weight](#1-the-gradient-of-a-masked-weight)
2. [The importance criterion](#2-the-importance-criterion) *(Phase 5)*
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

*Written in Phase 5.*

## 3. Where the engine bottlenecks

*Written in Phase 9.*

## 4. Serving a self-pruned model at scale

*Written in Phase 9.*
