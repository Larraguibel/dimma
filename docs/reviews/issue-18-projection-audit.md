# Pre-merge audit — projection mechanism (issue #18, PR #19)

Read-only audit of the projection-mechanism implementation on
`feat/issue-18-projection-mechanism`, run by three independent auditors
(two correctness, one quality/simplification) and spot-verified against the
actual code by the orchestrator. **No code was changed** — this is a record of
follow-up opportunities to implement later.

Scope: the ~724 added source lines across `core/projection.py`,
`core/noise.py`, `accounting/projection.py`, `mechanisms/projection.py`,
`algorithms/spiderboost/{kernels,train}.py`, `models/hashed_logreg.py`,
`models/losses.py`, and the touched `__init__.py` exports.

**Bottom line:** the DP math is correct and faithful to the paper, the Duchi
projection is provably right and jit-safe, and the `s=None` SpiderBoost path is
genuinely bit-exact. The items below are input-validation hardening and
behaviour-preserving cleanups — none block correctness of the happy path.

---

## Correctness findings

### C1 — Gaussian branch has no `ε < 1` guard (privacy caveat) — **verified**
`mechanisms/projection.py:107`, `accounting/projection.py:105`.
`gaussian_noise_scale` returns the classical Dwork–Roth calibration
`σ = √(2 ln(1.25/δ))·Δ₂/ε`, which is only valid for `ε ∈ (0, 1)`. This is
**faithful to the paper's stated formula** (Fact A.1), but the code accepts any
`ε > 0`; for `ε ≥ 1` the release under-noises and is *not* `(ε, δ)`-DP.
- Failure: `projection_mechanism(mean, epsilon=4.0, delta=1e-5, …)` returns a
  vector that does not satisfy `(4, 1e-5)`-DP.
- Options: (a) reject `ε ≥ 1` in the Gaussian branch with a clear error; or
  (b) adopt the analytic Gaussian mechanism (Balle–Wang 2018), valid for all ε,
  which is the more robust long-term fix. Either way, document the restriction.
- Note: this affects only the **standalone `projection_mechanism`**. The
  SpiderBoost integration reuses the existing, separately-calibrated
  `compute_noise_scales` and is unaffected.

### C2 — No `δ < 1` upper bound (silent no-noise / hard error) — **verified**
`mechanisms/projection.py:109` checks only `delta < 0.0`.
`gaussian_noise_scale` computes `√(8 ln(1.25/δ))`: at `δ = 1.25` the scale is
**0** (a "private" release with no noise); for `δ > 1.25`, `math.sqrt` of a
negative raises a cryptic `math domain error`; any `δ ≥ 1` is meaningless for
DP. Flagged independently by two auditors.
- Fix: require `0 ≤ δ < 1` (Gaussian branch strictly `0 < δ < 1`). One line.

### C3 — SpiderBoost kernels do not validate `s` (silent training freeze) — **verified**
`kernels.py` (anchor `radius = L0 * math.sqrt(s)`, variation
`radius = L1 * delta_w * math.sqrt(2 * s)`); `train.py` passes `config.s`
through unchecked.
- `s = 0` passes the `if s is None` guard → anchor radius `0.0` → every gradient
  estimate is projected to the **zero vector** → training silently freezes
  (finite, near-zero grad norms, no error).
- `s = -1` raises `math domain error` mid-`jit`, with no mention of `s`.
- `s = 2.5` / `s = True` accepted silently (type hint only).
- Inconsistent with the standalone mechanism, which eagerly validates `s ≥ 1`.
- Fix: eager guard in both factories (or in `train()` before jitting):
  `if s is not None and (not isinstance(s, int) or s < 1): raise ValueError(...)`.

### C4 — `hash_buckets` does not enforce its documented ≤ 2²⁴ bound — **verified**
`models/hashed_logreg.py` `hash_buckets`. The docstring states the `float32`
cast is exact only for indices `≤ 2**24`, but nothing asserts it. With, e.g.,
`num_buckets = 2**20`, indices exceed 2²⁴, the `float32` cast **aliases
buckets silently**, producing plausible-but-wrong gradients and violating the
exactness claim with no error.
- Fix: assert `num_fields * num_buckets <= 2**24` (or `max index ≤ 2**24`) in
  `hash_buckets` — cheap, eager, NumPy-side.

### Minor correctness (informational)
- **`project_l1_ball` returns the origin for `radius < 0`** silently (empty
  ℓ₁-ball). Unreachable from the mechanism (`radius = L√s > 0`); consider an
  `assert radius >= 0` in the public geometry function.
- **float32 `cumsum` precision** in `project_l1_ball` degrades at very large `d`
  (the high-dimensional regime this targets). Inherent to Duchi; already
  documented in the notebook. Consider dtype promotion if large-`d` accuracy
  matters.
- **Expected-vs-realized batch bias** (accounting/documented): averaging by the
  expected `b1`/`b2` while summing the realized Poisson batch can push the true
  pre-noise mean slightly outside `K`, an accepted `O(1/√b1)` distortion.
  Recorded in ADR-0003 — not a bug, listed for completeness.

**Verdicts:** DP math correct and paper-faithful; Duchi projection provably
correct and jit-safe with a traced radius; Laplace/pure-DP branch correct;
SpiderBoost radii, increment-before-accumulation placement, and post-processing
privacy claim all correct; `s=None` path bit-exact (original closures returned
verbatim, exercised by the regression oracle). The only real gaps are the
missing input guards C1–C4.

---

## Quality / simplification findings (behaviour-preserving unless noted)

Ranked by value. Line savings are approximate.

### Q1 — Deduplicate the projected kernel closures (~30 lines) — *verify with jaxpr diff*
`kernels.py`. Each projected closure re-types its base closure's body verbatim
(anchor 7/9 lines, variation 11/13). Because Python composition is invisible to
JAX tracing (same primitives, same order → identical jaxpr), the duplication can
be removed **without breaking the byte-identical `s=None` guarantee**:
- Anchor: `anchor_step_projected` calls `anchor_step(...)`, takes
  `.grad_estimate`, projects, rebuilds `StepOutput` (unused base norm is DCE'd
  under jit). ~9→4 lines.
- Variation: extract a shared
  `_noisy_increment(…) -> (noisy_delta, delta_w)` used by both variants
  (the base returns the *accumulated* estimate, so share the increment, not the
  whole step).
- **Verify first:** add a jaxpr/HLO-equality test proving the `s=None` program is
  unchanged. The refactor is trace-identical, but the module makes a hard
  bit-exact promise — prove it, don't assume it.

### Q2 — Extract the stable-BCE formula (single source of truth)
`models/losses.py`. The expression
`max(logit,0) − logit·y + log1p(exp(−|logit|))` is now written **three** times.
Extract a module-private `_stable_bce(logit, y)`; the three public losses become
one-liners. Net ~0 lines, but the subtle numerically-stable formula then lives
in exactly one place.

### Q3 — Factor the pytree-noise helper (~12 lines) — *verify pinned docstring*
`core/noise.py`. `add_pytree_laplace_noise` is a character-level copy of
`add_pytree_gaussian_noise` (`normal→laplace`, `std→scale`). Extract
`_add_pytree_noise(pytree, key, scale, sample_fn)`; both public functions become
2-line wrappers. **Verify first:** the module docstring pins the Gaussian
function as "extracted … without modification". The refactor is trace-identical,
but either update that sentence or *deliberately* keep the duplication and say so
in a comment.

### Q4 — Reconsider `return_noisy`'s polymorphic return — *API change, verify tests + notebook*
`mechanisms/projection.py`. The `bool → (T | tuple[T, T])` return is the
awkwardest shape in the diff and costs ~10 docstring lines. House style already
uses NamedTuples for multi-value returns (`StepOutput`, `TrainResult`); a
`ProjectionOutput(zhat, z_tilde)` (always returned) is cleaner and
self-describing. **Verify first:** updates the Lemma 3.1 tests and the demo
notebook that pass `return_noisy=True`.

### Minor quality / naming (low-risk)
- `core/projection.py`: rename `css` → `cssv`/`cumsum_u` (reads as a stylesheet);
  drop the redundant `rho.astype(css.dtype)` (int/float32 already promotes);
  annotate `radius: float | jax.Array` and the pytree variant's params/return to
  match `noise.py`'s house style.
- `mechanisms/projection.py`: annotate `key: jax.Array` (docstring says so, the
  signature is bare).
- `models/hashed_logreg.py`: promote the magic `0.01` init scale to a module
  constant `_INIT_STD` or an `init_scale` parameter.
- `accounting/projection.py`: the Laplace docstring's displayed math labels the
  scale `σ` while the prose insists it is the `b` parameter — rename the symbol
  to `b` for internal consistency. (Formula itself is correct.)
- `models/__init__.py`: exports both the `hashed_logreg` module *and* the
  `hashed_init_params`/`hashed_forward` aliases (two spellings of one API). The
  alias is a deliberate anti-shadowing choice; optionally drop the aliases and
  keep only the module (−4 export lines).

### Efficiency observations (likely no action)
- `projection_mechanism` runs eagerly (no jit) — fine for one-shot use; if a
  training loop ever calls it per step, jit the delta-branch bodies at the call
  site. One-line docstring note at most.
- `train.py` recomputes `delta_w` on the host after the kernel already computed
  it internally. Surfacing it via `StepOutput` would **break** the bit-exact
  `s=None` program, so the recompute is the correct trade-off — flagged only so
  nobody "optimizes" it later.

### Confirmed clean
`accounting/projection.py` (two tight pure functions; the `s`-asymmetry note is
genuinely useful), all three touched `__init__.py` files, the `train.py` `s`
field + wiring, `losses.py` (correct signature, calls the hashed forward), and
the layering itself — geometry in `core`, calibration in `accounting`, DP claim
in `mechanisms`, model in `models`, no layer violations.

---

## Suggested prioritization for the follow-up

1. **C2, C3, C4** — cheap, high-value input guards that close real footguns.
2. **C1** — decide guard-vs-analytic-Gaussian; it is the one privacy-correctness
   item, though it is faithful to the paper as written.
3. **Q2, Q1** — biggest clarity/line wins; Q1 needs the jaxpr-equality proof.
4. **Q3, Q4** — worthwhile but touch a pinned docstring / a public API; batch
   with a test update.
5. Minor naming/annotation nits — sweep opportunistically.
