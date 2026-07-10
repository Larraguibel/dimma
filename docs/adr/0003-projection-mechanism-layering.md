# ADR-0003: Projection mechanism — layering and SpiderBoost integration

**Status:** Accepted

## Context

Ghazi, Guzmán, Kamath, Kumar, Manurangsi, *"Differentially Private Optimization
with Sparse Gradients"* (NeurIPS 2024) introduces a perturb-then-project DP
mean-estimation primitive (Algorithm 1): add coordinate-wise noise to the
empirical mean of `s`-sparse, `L2 ≤ L` records, then Euclidean-project the noisy
answer onto the ℓ₁-ball `K = B₁(0, L√s)`. The projection is post-processing of a
private quantity, so it consumes no privacy budget, and it converts sparsity into
a **nearly dimension-independent** error rate — the error is driven by `‖ξ‖_∞`,
not `‖ξ‖₂` (Lemma 3.1). Each per-sample gradient in a DP optimizer is `s`-sparse
whenever the model has that structure (e.g. hashed categorical features), so the
primitive is a drop-in private gradient oracle (GitHub issue #18).

dimma has no DP-SGD module; the only training loop is Private SpiderBoost
(Algorithm 2 of Arora et al. 2023). We nonetheless want the projection primitives
to be **shared code** any future algorithm can compose unchanged, and we want the
SpiderBoost integration to be **default-off and bit-exact when off** so the
regression suite (`test_regression_against_source.py`) keeps pinning the original
kernels byte-for-byte. The paper's own bias-reduction machinery (Algorithms 2/3)
is substantial and orthogonal to this integration, and the existing package split
(`core/` makes no DP claims; `accounting/` calibrates noise; `algorithms/` runs
loops) has no home for a one-shot DP-claiming primitive.

## Decisions

### 1. A new `mechanisms/` package is the seam for one-shot DP-claiming primitives

The primitive is split across the existing layers by responsibility, with a new
package holding the DP-claiming composition:

- **Geometry** — `src/dimma/core/projection.py`: `project_l1_ball` (+ pytree
  variant), the Duchi et al. (2008) sort-based ℓ₁-ball projection. Names are
  geometric (`radius`), not sparsity-semantic. Makes **no** DP claim.
- **Noise** — `src/dimma/core/noise.py`: `add_pytree_laplace_noise`, mirroring the
  existing Gaussian helper. Makes no DP claim.
- **Calibration** — `src/dimma/accounting/projection.py`: the paper's exact scale
  formulas (`laplace_noise_scale`, `gaussian_noise_scale`), each tied to a named
  sensitivity of the empirical mean.
- **Mechanism** — `src/dimma/mechanisms/projection.py` (new package):
  `projection_mechanism`, Algorithm 1 end-to-end — calibrate, perturb, project.
  This is the layer that *claims* differential privacy.

**Why not put the mechanism in `core/`:** `core/` is deliberately DP-agnostic — it
holds reusable numerics (clipping, noise draws, pytree ops, projection geometry)
that make no privacy promise. A function that claims `(ε, δ)`-DP does not belong
there; putting it there would blur the "core makes no DP claims" invariant that
lets `core/` be tested purely numerically.

**Why not put it in `algorithms/`:** `algorithms/` is for training *loops* that
compose primitives over many steps. Algorithm 1 is a **one-shot** operation — the
exact thing a future DP-SGD would call once per step. Filing it under a specific
algorithm would make it look owned by that algorithm and invite a reach-in import
from the next one. `mechanisms/` names the missing seam: standalone,
DP-claiming, composable primitives that sit above `core/`/`accounting/` and below
`algorithms/`.

### 2. Default-off projection in both SpiderBoost kernels via a static factory arg

`make_anchor_step` and `make_variation_step` gain a single keyword argument
`s: int | None = None`. When `s is None` the factory returns the **existing
closure verbatim** — the same Python function object, hence the same traced/XLA
program — so `test_regression_against_source.py` stays bit-exact green (it is the
oracle that proves off-mode is a no-op). When `s` is an `int`, the factory returns
a *separate* projected closure. `s` is a **static** Python int captured at factory
time, so `sqrt(s)` (anchor) and `sqrt(2·s)` (variation) are compile-time
constants and the projection radius stays a clean traced expression.

The knob is surfaced as one **append-only** defaulted field on `TrainConfig`
(`s: int | None = None`, added after `margin_sigmas`). `TrainConfig` is a
`NamedTuple`; appending a trailing defaulted field keeps every keyword
construction site working and leaves the existing positional layout intact. `train`
wires it through with `make_anchor_step(fn, s=config.s)` /
`make_variation_step(fn, s=config.s)`.

**Why not a runtime `if project:` branch inside one kernel:** a data-dependent or
config-dependent branch inside the jitted step would either change the XLA program
for the off case (breaking bit-exactness) or require threading a traced flag
through the projection. A static factory arg keeps the two programs physically
distinct and makes "off" provably identical to the pre-projection kernel.

**Why append-only rather than reordering into a "logical" position:** positional
construction of a `NamedTuple` past `margin_sigmas` in downstream user code would
silently shift if we inserted the field mid-tuple. Appending is the only change
that is safe for both keyword and positional construction; all in-repo sites use
keywords, and this rule is recorded so future fields follow suit.

### 3. Paper-faithful radii, with the expected-batch-size distortion documented

The two radii instantiate Algorithm 1's `K = B₁(0, L√s)` for the two SpiderBoost
sensitivities:

- **Anchor step** — radius `L0·√s`. The anchor estimate is a mean of per-sample
  gradients clipped to `L2 ≤ L0`; a clipped s-sparse vector has
  `‖·‖₁ ≤ √s·‖·‖₂ ≤ L0·√s`. This is the direct Algorithm-1 analogue.
- **Variation step** — radius `L1·‖Δw‖·√(2s)`, applied to the noisy SPIDER
  **increment** `Δ_t` *before* the accumulation `∇_t = ∇_{t-1} + Δ_t`. The sparse
  object is the increment (a difference of two per-sample gradients at `w_t` and
  `w_{t-1}`), not the accumulated estimate. A difference of two s-sparse vectors
  is at most 2s-sparse and, once clipped to `L2 ≤ L1·‖Δw‖`, has
  `‖·‖₁ ≤ √(2s)·L1·‖Δw‖`.

**Expected-vs-realized batch-size caveat.** The kernels average by the *expected*
Poisson batch size (`b1`, `b2`), not the realized count — the standard DP
convention (an accounting requirement, see `CONTEXT.md → Poisson subsampling`).
The ℓ₁ bound above is derived for the true mean over the realized batch. When the
realized batch is smaller than expected, dividing by the larger expected size
shrinks the estimate (safe — it stays inside `K`); when the realized batch is
*larger* than expected, the true signal can sit slightly **outside** `K`, so the
projection can clip a small amount of genuine signal. This is an accepted
`O(1/√b1)` relative distortion (it vanishes as the batch grows and the realized
count concentrates on its mean); we do not correct it, because tracking the
realized count in the radius would either break the expected-size averaging
convention or make the radius depend on the sampling draw.

### 4. Bias of the projected estimator is accepted; bias-reduction is out of scope

Euclidean projection onto a convex set is a nonlinear contraction, so the
projected estimator is **biased** — `E[ẑ] ≠ z̄`. In the variation step the
per-increment bias telescopes within a phase (the increments accumulate into
`∇_t`, and so do their biases). The paper addresses this downstream with a
bias-reduction scheme (randomized exponentially-increasing batch sizes + a
telescoping estimator, Algorithm 2) wrapped in a randomly-stopped SGD (Algorithm
3). Those are **out of scope** for this issue: they are a different training loop,
not a post-processing tweak to SpiderBoost. We document the bias (kernel
docstrings, the demo notebook's caveats cell, and this ADR) rather than correct
it, so a consumer knows the projected run trades a small bias for the
near-dimension-independent variance reduction.

**Why not implement Algorithms 2/3 here:** they replace SpiderBoost's phase/anchor
structure with an adaptive-composition stopping rule and a bespoke batch schedule
— a new algorithm with its own accountant, not a projection option on the existing
one. Bundling them would couple two independent pieces of work and defeat the
"default-off, bit-exact when off" property this integration is built around.

### 5. Deferred: top-level re-export and a `resolve_config` passthrough for `s`

Two conveniences are intentionally **not** added yet:

- **Top-level `dimma.__init__` re-export of `projection_mechanism`.** The mechanism
  is reachable at `dimma.mechanisms.projection`; promoting it to the package root
  is a public-API commitment we defer until a second consumer (a real DP-SGD)
  exists to shape the surface.
- **`resolve_config` passthrough for `s`.** The Theorem B.3 resolver (ADR-0002)
  derives `{eta, q, b2}` and does not know about `s`; `s` is set directly on
  `TrainConfig`. A resolver passthrough (or a theory-driven default for `s`) waits
  until there is a principled rule for choosing `s`, which the paper frames as an
  open problem (`s` is a required, never-estimated input — see `CONTEXT.md →
  Gradient sparsity`).

## Consequences

- The projection primitives are shared across layers and reusable by a future
  DP-SGD unchanged; the "common code" requirement of issue #18 is met without a
  DP-SGD module existing today.
- Off-mode (`s = None`) is provably a no-op: same closure, same XLA program, so
  `test_regression_against_source.py` continues to pin the original kernels
  byte-for-byte.
- Privacy accounting is untouched. `compute_noise_scales`, `NoiseScales`,
  `StepInfo`, `TrainHistory`, and `resolve_config` are unchanged; turning
  projection on does not alter the `(ε, δ)` claim, only the estimator's
  bias/variance trade-off.
- Callers accept a biased estimator when `s` is set. Any future need for an
  unbiased sparse gradient oracle must implement the paper's bias-reduction
  (Algorithms 2/3) as a *new* algorithm, at which point the deferred top-level
  re-export and an `s`-aware config resolver should be revisited.
- `TrainConfig` is now append-only by rule: new fields go at the end with a
  default, preserving positional and keyword construction.
