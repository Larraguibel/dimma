# ADR-0002: Theorem B.3 config resolver — deriving SpiderBoost hyperparameters

**Status:** Accepted

## Context

The example notebooks (notably `theorem42_stationarity_rates`) set the Private
SpiderBoost hyperparameters `eta`, `q`, and `b2` by hand (e.g. `q=30`, `eta=0.01`,
`b2=512`), independent of the privacy budget. These are not free knobs: Theorem
B.3 of Arora et al. 2023 ("Faster Rates of Convergence to Stationary Points in
Differentially Private Optimization", PDF in `base_papers/`) prescribes each as a
function of `ε, δ, L0, L1, n, d, T, F0`. Setting them arbitrarily means the runs
do not trace the algorithm instance the paper analyses (GitHub issue #7).

We want the library — not each notebook — to own this derivation, so every
consumer gets the proof-prescribed instance by default while retaining the
ability to override any value.

`TrainConfig` deliberately holds algorithm parameters only; it does not know `n`,
`d`, or `F0`. And `compute_noise_scales` (a separate, pre-`train` step) already
consumes `q`, `T`, `b2`. So derivation must happen *before* `compute_noise_scales`
and needs inputs `TrainConfig` does not carry.

## Decisions

### 1. A pure resolver function, not a change to TrainConfig or train

Add `resolve_config(init_params, n, F0, *, epsilon, delta, L0, L1, T, b1, seed,
q=None, b2=None, eta=None, ...) -> TrainConfig` in
`src/dimma/algorithms/spiderboost/config.py`. It returns a fully-concrete
`TrainConfig`. `TrainConfig`, `compute_noise_scales`, and `train` are unchanged —
the resolver is a new layer mirroring the existing "call `compute_noise_scales`
before `train`" idiom.

**Why not Optional fields on TrainConfig + a `.derive()` method:** that creates
two states (raw vs resolved) of the same type, inviting an unresolved config to
reach `train`. Keeping `TrainConfig` always-valid is safer.

**Why not derive inside `train`:** `compute_noise_scales` runs earlier and needs
the derived `q`; deriving inside `train` would be too late.

### 2. `None` means "derive from theory"; explicit means "use as given"

Any of `q`, `b2`, `eta` left `None` is filled from Theorem B.3. An explicit value
is respected. A missing *non-derivable* mandatory parameter raises, as usual.
`d` is computed from `init_params` (`sum(leaf.size …)`); `F0` is passed by the
caller (initial suboptimality `F(w₀;S)`, `F*=0` convention).

Derived quantities (verified against Theorem B.3, p. 16):

```
eta = 1 / (2·L1)
q   = floor( n²·ε² / (L1²·T·d·log(1/δ)) )
b2  = floor( max{ (L0·n·ε / sqrt(F0·L1·d·log(1/δ)))^(2/3),
                  (L0·n·d·log(1/δ))^(1/3) / ((L1·F0)^(1/6)·ε^(2/3)) } )
```

### 3. Follow the theorem *statement* for `q`, not the proof

The paper's Theorem B.3 *statement* keeps the `L1²` in the `q` denominator. Its
*proof* (p. 17) sets `q = 1/(T·ᾱ²) = n²ε²/(T·d·log(1/δ))` — without `L1²`, because
`η=1/(2L1)` cancels it. The two disagree by a factor of `L1²` (= 25× at `L1=5`).
We follow the **statement**: it is the formal claim, it yields sensible `q`
values, and issue #7's validated numeric table matches it. The discrepancy is
noted in the resolver docstring.

### 4. `T` and `b1` stay caller-provided, not derived

Both are derivable from Theorem B.3 but take impractical values for the Criteo
setting (`T` ~ tens of thousands; `b1 = n`, the full-batch anchor). They remain
required inputs, documented as off-theory budget choices. The auto-derived set is
exactly `{eta, q, b2}`.

### 5. Strict in-regime enforcement

The resolver lower-guards `q = max(1, q)` and **raises `ValueError`** when the
derived instance leaves Theorem B.3's stated regime:

- `q > T` — phase length exceeds the horizon (the anchor never refreshes).
- `b2 > n` — expected Poisson batch larger than the dataset (ill-defined).
- `n < n_min`, where
  `n_min = max{ (L0·ε)²/(F0·L1·d·log(1/δ)),  sqrt(d)·max{1, sqrt(L1·F0)/L0}/ε }`.

Rationale: the resolver's purpose is to instantiate *the* proof-prescribed
instance; building one the proof does not cover would be silently misleading. A
consequence is that with `T=200` only `ε ≲ 0.26` is in-regime; running `ε=2`
in-proof needs `T ≈ 1500`. Callers must raise `T` or lower `ε` accordingly.

### 6. Report provenance

The resolver prints which parameters were derived and which inputs each used
(values, not formulas — formulas live in this ADR and the docstring), so a reader
can audit the instance at a glance.

### 7. `L0`/`L1` are documented design inputs, never estimated

`L0` is the per-sample gradient clipping threshold; `L1` is the smoothness
constant used in the variation-step sensitivity. For an MLP with unbounded
weights neither has a finite global value over parameter space (the reason
DP-SGD clips), and input-space estimators (LipSDP, spectral-norm products) bound
the wrong object (Lipschitzness w.r.t. inputs, not parameters). The library
therefore requires them as inputs and documents them; it provides no estimator.

## Consequences

- Notebooks call `resolve_config(...)` instead of hand-setting `eta`/`q`/`b2`;
  the derivation is testable in isolation (pure function of scalars + a pytree).
- Sweeps must stay in-regime or the resolver raises — surfacing, rather than
  hiding, the `T=200` vs large-`ε` tension that issue #7 set out to expose.
- If a future change derives `T` (and thus relaxes the fixed-`T` limitation),
  the `q ≤ T` interaction and this ADR should be revisited.
- `b2` inherits `F0`'s approximation; callers should treat derived `b2` as
  exact-given-`F0`, not exact-absolutely.
