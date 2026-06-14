# Implementation notes (heuristics, deferred decisions)

!!! note

    These are implementation choices in dimma's Private Spider-Boost that don't show up in the paper. Each is a deferred or heuristic decision worth understanding before changing default parameters or trusting the realised epsilon.

    Implementation: `dimma.algorithms.spiderboost`, `dimma.accounting.spiderboost`.

## The constant `c` in the noise scales

Algorithm 2 of Arora et al. presents the noise scales as

```js
σ_1     = c * L_0 * sqrt(log(1/δ)) / ε * max(1/b_1, sqrt(T)/(q*n))
σ_2     = c * L_1 * sqrt(log(1/δ)) / ε * max(1/b_2, sqrt(T)/n)
σ̂_2     = 2c * L_0 * sqrt(log(1/δ)) / ε * max(1/b_2, sqrt(T)/n)
```

where `c` is described as a "universal constant" from the privacy proof (Theorem B.2). The paper does not give a numerical value. dimma defaults `c = 1.0`, inherited from the original implementation. This is **a heuristic, not a derivation**.

What to do about it. Treating `c = 1.0` as conservative or anti-conservative would require unfolding the constants in the proof of Theorem B.2, which has not been done. For now: the default is exposed as `DEFAULT_C` in `dimma.accounting.spiderboost`, callers can override it, and any privacy claim made with `c = 1.0` should be sanity-checked against an RDP accountant (`verify_epsilon`) before being reported.

## The `T+1` iteration count

The algorithm box of Arora et al. reads:

```js
for t = 0, ..., T do
  ...
  w_{t+1} = w_t − η∇_t
end for
return w̄ uniformly at random from {w_1, ..., w_T}
```

So the loop runs **T+1 iterations** (t = 0, 1, ..., T), produces iterates w_1, ..., w_{T+1}, and returns one of `{w_1, ..., w_T}`. The current dimma implementation matches this exactly.

There is, however, a subtler accounting question. The privacy proof (Theorem B.2) sums sensitivities over `t ∈ [T]`, which in the paper's notation is `{1, ..., T}` — i.e., **T queries, not T+1**. The noise formula uses `sqrt(T)`. So the loop runs T+1 mechanism invocations while the calibrated noise is sized for T. Either the paper is implicitly assuming step 0 contributes nothing (it doesn't — step 0 is an anchor step), or there's a small off-by-one in the proof's bookkeeping.

dimma preserves the paper-literal loop. `verify_epsilon` counts T+1 mechanism invocations, internally consistent with the loop. The cost of the discrepancy is one extra Gaussian mechanism, which is negligible in magnitude but worth flagging.

!!! warning

    **Status:** open. Should be resolved by re-reading Theorem B.2 carefully or by asking the authors.

## The variation-step sensitivity bound in `verify_epsilon`

The variation step in Algorithm 2 clips each per-sample gradient difference to `L_1 * ‖w_t − w_{t-1}‖`, where `L_1` is the gradient-Lipschitz constant. The realised per-sample sensitivity is therefore `2 * L_1 * ‖w_t − w_{t-1}‖` (each of the two clipped per-sample gradients contributes once to the difference).

For the RDP sanity-check in `verify_epsilon`, dimma uses the **noise cap `σ̂_2`** to upper-bound the actual noise standard deviation, and converts to a noise multiplier as `z = σ̂_2 * b_2 / (2 * L_0)`. The `2 * L_0` upper-bounds `2 * L_1 * ‖w_t − w_{t-1}‖` whenever the iterates move enough that the cap binds. This is **loose** when the iterates are settled and the cap doesn't bind; in that regime the realised sensitivity is `2 * L_1 * ‖w_t − w_{t-1}‖ ≪ 2 * L_0`, and `verify_epsilon` overestimates the privacy cost.

The loose bound is the right thing to do for a sanity check (over-estimating cost is safe). The paper's analysis uses the tighter sensitivity directly and is what gives Theorem 4.2 its rate. Don't use `verify_epsilon`'s output as the headline privacy guarantee — use the paper's bound.

## Truncated Poisson sampling and its accountant

The original Criteo implementation pads Poisson-sampled batches to a fixed `b_max` and truncates the (very rare) oversize draws to `b_max` slots. This is convenient for JIT, but it changes the mechanism: the standard Poisson-subsampled Gaussian privacy bound assumes batches are *not* deterministically truncated.

dimma exposes both variants explicitly:

- `poisson_subsample` — strict; raises on oversize draws.
- `poisson_subsample_truncated` — truncates deterministically.

The matching accountants (`poisson_gaussian_epsilon`, `poisson_gaussian_truncated_epsilon`) are also distinct. The truncated accountant returns the **same** numerical value as the strict one in the current implementation, but with a docstring stating clearly that this is a lower bound on the true privacy cost of the truncated mechanism. Tightening this bound is an open research question; until it's resolved, treat the truncated variant as research-grade and prefer the strict variant for runs whose privacy claim matters.

!!! warning

    **Practical note:** with the default `margin_sigmas=6` in `poisson_padded_batch_size`, the oversize probability is around 10⁻⁹ per step. For typical T this means the truncated and strict samplers produce identical outputs in practice, and the choice between them is about which accounting story you want to publish, not about realised behaviour.

## RNG consumption asymmetry on truncation events

`poisson_subsample_truncated` consumes additional numpy RNG state on oversize events (it calls `rng.choice(idx, size=b_max, replace=False)`). The strict sampler does not. This means **the RNG state at step `t+1` depends on whether step `t` triggered truncation**.

Consequence: a run that triggers a truncation event has a different Poisson-mask sequence from that point onward than the otherwise-identical run that did not. Reproducibility per fixed seed is preserved — the same seed and same data give the same sequence of events including truncations. But you cannot compare two runs that differ only in whether one happened to truncate; their masks diverge.

Not a correctness issue. Worth knowing if you're doing fine-grained ablations or RNG audits.

## F₀ in Theorem 4.2 is not directly computable

Theorem 4.2 uses **F₀ = F(w₀; S) − min_w F(w; S)**, the initial suboptimality relative to the global empirical minimum. For a non-convex model such as the Criteo MLP, the global minimum is unknown, so F₀ cannot be computed exactly.

Three practical approximations exist:

1. **F(w₀; S) directly** — a valid upper bound, since F* ≥ 0 implies F₀ ≤ F(w₀; S). Safe and defensible; always inflates the theoretical bound upward, consistent with O() being an upper bound.
2. **F(w₀; S) − F(w_T; S)** — uses the final training loss as a lower bound on F*. Tighter, but requires a completed run, and F(w_T; S) may still be far above the true minimum.
3. **F(w₀; S) − F_bayes** — subtracts an estimated irreducible noise floor (≈ 0.44–0.45 nats for Criteo). Dataset-specific and informal.

**Why the choice rarely matters in practice.** F₀ appears in term 1 of the bound only as F₀^{1/3}. A factor-of-2 error in F₀ produces a 2^{1/3} ≈ 1.26× error in term 1. When fitting the unknown constant C empirically, this imprecision is absorbed into C and does not affect the ε-dependence being tested.

**Recommendation:** use option 1 (plain initial loss) and state it clearly. Reserve option 3 for sensitivity checks.

## Loose ends

Things noted during the dimma build that deserve attention but were not fixed in the initial extraction:

- **`pytree_global_norm`** uses Python's `sum()` over a generator of `jnp.sum` scalars rather than a tree-reduce. Functionally equivalent, slightly less JIT-friendly. Candidate for a perf pass once a model large enough to make this matter is involved.
- **Host-side recomputation of `Δ_w`** in the variation branch: the training loop recomputes `‖w_t − w_{t-1}‖` on the host for the `StepInfo` callback, even though the kernel computed it internally. Cheap for small models. The clean fix is to extend `StepOutput` to return `Δ_w` from the kernel; deferred.
