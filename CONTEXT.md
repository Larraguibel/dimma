# dimma — Domain Language

This file is the **ubiquitous language glossary** for the dimma project.
One term per concept. Use these names — and only these names — in code, comments, issues, and prompts.
No implementation details, no specs, no scratch pad.

---

## Privacy guarantees

**Differential Privacy (DP)**
A formal mathematical guarantee that the output of an algorithm changes negligibly when any single individual's data is added or removed. Parameterized by (ε, δ): ε bounds the privacy loss; δ bounds the probability of a catastrophic failure.
_Avoid_: data privacy, anonymization, privacy guarantee, approximate DP

**Privacy budget**
The total allowable (ε, δ) expenditure across all training rounds. Once consumed, no further training is permitted under the same guarantee.
_Avoid_: privacy cost, epsilon budget, DP budget

**Rényi Differential Privacy (RDP)**
An intermediate privacy accounting framework (parameterized by order α) that composes tightly across rounds and converts to (ε, δ)-DP at the end. Used internally by `compute_noise_scales`.
_Avoid_: moment accountant, RDP accounting, Rényi DP

**Privacy accounting**
The process of tracking cumulative privacy loss across training steps. In dimma this is delegated to Google's `dp-accounting` library.
_Avoid_: privacy tracking, epsilon tracking

---

## Training structure

**Phase**
One complete cycle of one anchor step followed by `q` variation steps. The unit of iteration in Private SpiderBoost.
_Avoid_: epoch, round, cycle

**Phase length (q)**
The number of variation steps per anchor step within one phase. A key hyperparameter: larger `q` amortizes the cost of the anchor step but increases variance accumulation.
_Avoid_: steps per phase, inner loop count, q parameter

**Anchor step**
The step within a phase that computes a full gradient at the current reference point, establishing the variance-reduction baseline. Corresponds to the SPIDER "snapshot" computation.
_Avoid_: snapshot step, reference computation, full-batch step, outer step

**Variation step**
Each of the `q` lightweight steps within a phase that uses a stochastic gradient estimator corrected by the anchor-step baseline. Cheaper per step than an anchor step.
_Avoid_: inner step, correction step, SPIDER step

**Final iterate**
The model parameters at the end of the last variation step — the last point in the trajectory. Returned as `TrainResult.params_final`.
_Avoid_: last checkpoint, terminal parameters

**Random iterate**
A uniformly sampled intermediate iterate from the training trajectory. This is the iterate with the formal (ε, δ)-DP convergence guarantee. Returned as `TrainResult.params_random`.
_Avoid_: random checkpoint, sampled params, output with guarantee

---

## Noise and clipping

**Noise scale (σ)**
A scalar multiplier on the Gaussian noise added to a gradient. dimma uses three: `σ₁` (anchor step), `σ₂` (variation step), `σ₂_hat` (bias correction term). Calibrated by `compute_noise_scales` to satisfy the privacy budget.
_Avoid_: noise multiplier, sigma, standard deviation

**Per-sample gradient clipping**
Bounding each individual sample's gradient to L2-norm ≤ C before aggregation. Required for DP: without it, a single sample could dominate the update.
_Avoid_: gradient clipping, norm clipping, clip

**Clipping threshold (C)**
The maximum L2 norm allowed for a per-sample gradient. Denoted `L0` (anchor) and `L1` (variation) in the API, following the paper's notation.
_Avoid_: clip norm, max gradient norm

---

## Sampling

**Poisson subsampling**
Selecting each training example independently with probability `q` (the sampling rate), rather than drawing a fixed-size batch. This amplifies the privacy guarantee multiplicatively.
_Avoid_: random sampling, batch sampling, q-sampling

**Sampling rate (q)**
The per-example inclusion probability for Poisson subsampling. Not to be confused with phase length `q` — they share notation in the paper but are the same hyperparameter (they are linked in the `TrainConfig`).
_Avoid_: batch fraction, subsample probability

---

## Model interface

**Pytree**
A JAX-native nested structure of arrays (dicts, lists, tuples of arrays) used to represent model parameters and gradients. dimma operates on arbitrary pytrees — it is model-agnostic.
_Avoid_: parameter dict, weight tensors, model weights

**Per-sample loss function**
A callable `(params, x_i, y_i) → scalar` that computes the loss for a single example. The user provides this; dimma applies `jax.vmap` internally to vectorize over the batch.
_Avoid_: loss function, batch loss, objective

---

## Public API surface

**`train()`**
The single entry point for running Private SpiderBoost. Takes data, a per-sample loss function, initial params, a config, and noise scales. Returns a `TrainResult`.

**`TrainConfig`**
A dataclass holding all hyperparameters: ε, δ, L0, L1, T (total phases), q (phase length / sampling rate), b1 (anchor batch size), b2 (variation batch size), η (learning rate), seed.

**`TrainResult`**
The output of `train()`: contains `params_final`, `params_random`, and `history`.

**`compute_noise_scales()`**
Calibrates the three noise scales (σ₁, σ₂, σ₂_hat) from the privacy budget and training config using RDP accounting. Must be called before `train()`.

**`NoiseScales`**
A named triple `(σ₁, σ₂, σ₂_hat)` returned by `compute_noise_scales()`.
