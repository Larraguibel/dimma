# dimma: the library

!!! note

    **dimma** is the JAX library that codifies the DP-SGD patterns from [Jax, Flax and Other DP-SGD Libraries](tooling.md) into reusable primitives for non-standard differentially private optimization. This page covers what's in it, how it's structured, the design decisions worth knowing, and how to add a new algorithm.

    Repository: [https://github.com/Larraguibel/dimma](https://github.com/Larraguibel/dimma) (private as of writing)

## What dimma is, and what it isn't

The project has two layers of documentation, and dimma sits between them. The conceptual material in [Differential Privacy for SGD: Overview](overview.md) explains *what* DP-SGD is. The tooling material in [Jax, Flax and Other DP-SGD Libraries](tooling.md) explains *how* you'd write one in JAX from primitives. **dimma is the next step**: a small library that codifies those primitives into a stable, reusable, dataset- and model-agnostic surface, so each new algorithm is a thin specialisation rather than a fresh from-scratch implementation.

It is **not** a standard ML training framework. The training loop is hand-written JAX, exposed for inspection. There is no `model.fit(...)`, no Keras-style abstraction, no Optax-style aggregated-gradient pipeline. Each algorithm in dimma plugs into the same set of primitives (per-sample clipping, Gaussian noise, Poisson subsampling) and exposes its own training entry point.

It is also **not** an accounting framework. Accounting is injected per algorithm. Where standard sampling-based accounting applies, dimma provides it (`dimma.accounting.sampling.poisson_gaussian_epsilon` wraps `dp_accounting`'s RDP). Where the algorithm uses non-standard accounting (Spider-Boost's `compute_noise_scales`), the library exposes that too.

## Module map

The top-level surface is intentionally small. The user-facing entry point is `dimma.train` for the Spider-Boost algorithm; everything else sits under submodules.

```js
dimma/
├── train, TrainConfig, TrainHistory, TrainResult, StepInfo   ← Spider-Boost entry
├── NoiseScales, compute_noise_scales                          ← Spider-Boost accounting
│
├── core/         ← shared primitives across algorithms
│   ├── pytree     (norm, sub, add, scale, zeros_like, ...)
│   ├── clipping   (per_sample_norms, per_sample_clip, per_sample_apply_mask)
│   ├── noise      (add_pytree_gaussian_noise)
│   └── sampling   (poisson_subsample, poisson_subsample_truncated, padded_batch_size)
│
├── algorithms/
│   └── spiderboost/
│       ├── kernels       (make_anchor_step, make_variation_step, sgd_update)
│       └── train         (the outer loop, generic over loss and init_params)
│
├── accounting/
│   ├── spiderboost  (NoiseScales, compute_noise_scales, verify_epsilon)
│   └── sampling     (poisson_gaussian_epsilon, poisson_gaussian_truncated_epsilon)
│
├── datasets/     ← optional convenience layer (NOT depended on by algorithms)
│   ├── base       (TabularSplit, arrays_to_split)
│   └── criteo     (load_criteo)
│
└── utils/
    └── device     (resolve_device)
```

**Why this split.** `core/` holds the primitives that every per-sample DP algorithm needs. `algorithms/` is where each algorithm lives, isolated. `accounting/` is split the same way: generic sampling-based accountants are reusable, algorithm-specific accountants live alongside their algorithm. `datasets/` is convenience scaffolding; the algorithms never import from it. `utils/` is for genuinely dataset- and algorithm-agnostic helpers.

Note `core/` is not promoted to the top namespace. `dimma.core.clipping.per_sample_clip` is the path; this is deliberate. The top-level surface is what users *call*, not the internals.

## Design conventions worth knowing

These are decisions that look like style but are load-bearing. Each one is here because doing the opposite would either compromise DP correctness or set a bad precedent for future algorithms.

### Per-sample primitives live in `core/`, not in `algorithms/`

`per_sample_clip`, `per_sample_norms`, `per_sample_apply_mask`, `add_pytree_gaussian_noise` are reusable across **every** per-sample DP algorithm. They are the primitives. `algorithms/` modules only orchestrate calls into them, never reimplement them. If you find yourself writing per-sample clipping logic inside an algorithm module, you are duplicating `core/`. Stop and reuse.

### Losses are required arguments, not defaulted

`dimma.train` requires `per_sample_loss_fn` and `init_params` as explicit arguments. There are no model defaults. This is annoying for tutorial code (you have to write three more lines) and exactly right for a DP library: the call site explicitly names the loss function being privatised, which is the function whose per-sample gradient is being clipped. If a default existed, a user who forgot to pass it would silently train on whatever the default loss assumed about the data, and not realise until much later.

### Evaluation lives outside the training loop

The library returns `TrainResult` (final params, random-output params, history). It does not compute AUC, log to wandb, or print progress. The caller passes a `step_callback(StepInfo) -> None` to do whatever per-step bookkeeping they want, and computes metrics on `result.params_final` / `result.params_random` after the run.

This is partly separation of concerns (the library doesn't know what metric you care about) and partly correctness: any evaluation that touches the training data is itself a privacy-relevant operation. Forcing the caller to write it explicitly means they have to think about whether it is.

### Two Poisson samplers, named distinctly

The library ships `poisson_subsample` (strict; raises on oversize batches) and `poisson_subsample_truncated` (deterministically truncates oversize batches). They exist as separate functions, not as one function with a flag, because the privacy story differs:

- **`poisson_subsample`** matches the standard Poisson-subsampled Gaussian mechanism. Its accountant (`poisson_gaussian_epsilon`) uses the standard `dp_accounting.PoissonSampledDpEvent` bound. No surprises.
- **`poisson_subsample_truncated`** is a *modified mechanism*. Truncation introduces adversary advantage on the rare oversize-batch event. Its accountant (`poisson_gaussian_truncated_epsilon`) returns the standard bound, but **as a lower bound on the true privacy cost**. The docstrings say this loudly. Tightening this bound is an open research question; we do not pretend it is solved.

Call sites are greppable: `git grep poisson_gaussian_truncated_epsilon` finds every place that relies on the heuristic.

### `step_callback` exposes `StepInfo`, not raw batches

The callback receives a `StepInfo` containing the step index, params, grad estimate, grad norm, `delta_w`, and realised noise std. It **does not** receive `x_batch` or `y_batch`. This is a deliberate restriction. The batch data is what the privacy mechanism is protecting; handing it to user code at every step would create one more surface for accidental leakage.

The practical consequence: the per-step training-loss plot from the original Criteo notebook cannot be reproduced exactly through the callback. If you need it badly enough, the right library-level fix is for `train` to accept an optional `batch_loss_fn(params, x, y) -> scalar` and compute the loss internally before calling the callback. That keeps raw data inside the loop. Not implemented yet; flagged as a design question if it becomes important.

### RNG separation: privacy-relevant vs control flow

The training loop derives three RNGs from `config.seed`:

- `sampling_rng` (numpy.Generator, seed) — drives Poisson masks. Privacy-relevant.
- `noise_key` (jax.random.PRNGKey, seed+1) — drives Gaussian noise. Privacy-relevant.
- `control_rng` (numpy.Generator, seed+7919) — draws the random output step. **Not** privacy-relevant.

Keeping the output-step draw in its own RNG means the two privacy-relevant streams never get perturbed by control-flow randomness. It also means runs with the same seed but different `T` produce identical Poisson masks for the steps they share, which is useful for debugging.

One subtlety: `poisson_subsample_truncated` consumes additional `sampling_rng` state on oversize events (it calls `rng.choice` to subsample). This means the RNG state at step `t+1` depends on whether step `t` triggered truncation. Reproducibility per fixed seed is preserved; cross-run RNG state is not.

## The `datasets/` module: convenience, not a contract

`dimma.datasets.load_criteo()` downloads the 1M Criteo sample from Hugging Face (CC-BY-NC-SA 4.0, attribution printed on first download) and returns a `TabularSplit`. It is provided so that incoming students can run their first end-to-end DP-SGD in three lines. The library's algorithms do **not** import from `dimma.datasets` — if you delete the module, dimma still works.

!!! warning

    **The Criteo parquet on HF has hidden preprocessing.** The columns `I1..I13` are documented as integer features with missing values, but the uploaded parquet has already been min-max scaled to `[0, 1]` and contains no NaNs. The library's `features="integer"` mode then applies log1p + per-feature standardisation on top, which is mathematically valid but operates on already-normalised data. The `features="all"` mode returns the raw parquet content untouched. If you want true raw integer-with-NaN semantics for Criteo, you'd need to re-derive from the original Kaggle dump.

## How to add a new algorithm

The library is set up so a second algorithm is a self-contained addition, not a refactor. The rough recipe:

1. **Create `dimma/src/dimma/algorithms/<name>/`** with `__init__.py`, `kernels.py`, and `train.py` (matching the Spider-Boost layout).
2. **Write the per-step kernels in `kernels.py`.** They should be pure functions over a parameter pytree. Use the primitives in `dimma.core` (`per_sample_clip`, `per_sample_apply_mask`, `add_pytree_gaussian_noise`, the pytree helpers). Do not reimplement these. If your algorithm needs a primitive that doesn't exist, add it to `dimma.core` first, with tests, and only then use it from your algorithm.
3. **Write the training loop in `train.py`.** Follow the Spider-Boost pattern: a `TrainConfig`, a `StepInfo`-like NamedTuple if your algorithm has interesting per-step state, a `TrainResult`. The loop takes `per_sample_loss_fn` and `init_params` as required arguments. It uses `dimma.core.sampling` for subsampling. It does not do evaluation or printing.
4. **Write an accounting module under `dimma/accounting/<name>.py`** (or place generic sampling-based accounting in `dimma.accounting.sampling` if it already covers your case). Algorithm-specific accountants belong with their algorithm.
5. **Test against a synthetic problem.** A 2-parameter linear model on a few dozen examples is enough. Verify the kernels do what they're supposed to with `sigma=0` (deterministic baseline) and with the actual noise.
6. **Document the paper-vs-implementation differences** under the algorithm's Notion page, mirroring [Differences between theory and implementation](algorithms/spiderboost/theory-vs-implementation.md) for Spider-Boost.
7. **Do not** add the algorithm to `dimma.__init__.py` until the public API is stable. Keep it accessible at `dimma.algorithms.<name>.train` until you're confident in the surface.

The second algorithm is the real test of whether `core/` was structured correctly. If you find yourself reimplementing primitives or copy-pasting from `algorithms/spiderboost/`, the abstraction is wrong and `core/` needs to grow. That's expected and fine — it's why the project's design principle is "implement two concrete algorithms fully before extracting shared abstractions."

## Open verification items

Things that are *not* known to be wrong but are also not yet pinned down:

- **Full-T numerical reproduction.** The adapted Criteo notebooks (T=200) run in a few seconds and produce AUCs around 0.64; the original (T=1500) ran for ~20 minutes and reached 0.76–0.79. The library is functionally equivalent at short T, but a one-shot full-T run to pin numerical equivalence against the original is still pending.
- **Second algorithm.** Until one exists, `core/` could be over-fit to Spider-Boost without us knowing.
- **Tightening the truncated-Poisson accountant.** Currently returns the standard bound as a lower bound on the true cost. A real analysis is an open research question.

## Pointers

- DP background: [Differential Privacy for SGD: Overview](overview.md)
- JAX / Flax / Optax patterns the library codifies: [Jax, Flax and Other DP-SGD Libraries](tooling.md)
- First algorithm implemented in dimma: [Private Spider-Boost](algorithms/spiderboost/index.md)
