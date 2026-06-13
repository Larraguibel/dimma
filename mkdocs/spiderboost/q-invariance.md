# The q-invariance of params_random

!!! note

    A non-obvious algorithmic property of the Spider-Boost output rule. Worth knowing so q-sweep results aren't over-interpreted.

## The property

The Spider-Boost output rule returns `w̄`, an iterate sampled uniformly from `{w_1, ..., w_T}`. dimma implements this by drawing the output index `t*` from a *separate* RNG stream (`control_rng`, seeded as `config.seed + 7919`), at the start of the run, **before** any training step.

This means: **for any two runs that share `config.seed` but differ only in `q`, the value of `t*` is identical.**

Now consider what `w_{t*}` looks like as a function of `q`. The trajectory `w_1, ..., w_{t*}` depends on the sequence of anchor vs. variation steps up to step `t*`. Anchor steps fire at `t = 0, q, 2q, ...`. So:

- For any `q` such that the **next anchor after `t = 0`** lands at or after `t*`, the trajectory `w_0, w_1, ..., w_{t*}` is identical across those `q` values. It consists of one anchor step at `t = 0` followed by `t* − 1` variation steps, with no other anchor.
- The Poisson masks for those steps are identical (same `sampling_rng`, same step count, same expected batch sizes). The noise keys are identical (same `noise_key` split chain). The per-sample gradients are identical (same params, same batches). So `w_{t*}` is *bit-identical* across all such `q`.

The result: `params_random` is q-invariant on the set of `q` values where the first non-zero anchor (`t = q`) lands at or after `t*`.

## When this bites

The q-sweep notebook in `dimma/examples/criteo/` runs with `T = 200`. The output step `t*` came out to 82. For that run:

| q | first non-zero anchor | identical to other q? |
| --- | --- | --- |
| 10 | t = 10 | no (anchor before t\*) |
| 30 | t = 30 | no (anchor before t\*) |
| 100 | t = 100 | yes |
| 200 | t = 200 | yes |

`q = 100` and `q = 200` both have their first non-zero anchor at or after `t* = 82`, so `params_random` (and therefore `AUC(w̄)`) is bit-identical between the two. This was first observed as an unexplained 0.6094 AUC collision in the q-sweep; the diagnostic script in `dimma/examples/criteo/diagnose_q_collision.py` confirmed `max|Δ params_random| = 0.0`.

`params_final = w_T` is **not** subject to this invariance — by step T the trajectories have always diverged, so the final-iterate AUC genuinely depends on `q`.

## Implication for q-sweeps

For short runs where `t*` is small relative to the `q` range being swept, the random-iterate AUC curve can have **flat segments** that aren't measuring anything about `q`. They're an algorithmic invariant.

Two ways to interpret an apparently flat AUC(w̄) at large `q`:

1. **Genuine flatness in the privacy-utility tradeoff** — the algorithm performs the same across that range of `q`.
2. **The invariance above** — the runs literally produced the same iterate.

To distinguish, compute `output_step` for each run. If they all share the same `t*`, and the smallest `q` in the flat segment is `> t*`, you're seeing (2). If the `q` values straddle `t*` and you still see flatness, you're seeing (1).

For long-T runs the issue largely disappears: `t*` is uniformly distributed over `{1, ..., T}`, so most draws land well after `q`, where the trajectories have diverged. The invariance is a short-T artifact in the regime where `t* < q` is plausible.

## Is this a bug?

No. The output rule samples uniformly from `{w_1, ..., w_T}` regardless of `q` — that's the algorithm. The invariance is what happens when you sample at an index where the relevant `q`-dependent computation hasn't yet occurred. The diagnostic confirmed the trajectories diverge after `t = 100` (max abs diff in `params_final` was ~0.08), so the algorithm is working correctly; it's just that the random output rule happened to look at the trajectory before that divergence.

The right response is to know the property exists and to read q-sweep plots with it in mind. No library change is needed.
