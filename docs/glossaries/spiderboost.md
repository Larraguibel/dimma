# Private SpiderBoost — Glossary

Vocabulary specific to the Private SpiderBoost algorithm (Arora et al., ICML 2023).
This is the canonical naming contract for SpiderBoost code, comments, issues, and
prompts — the same role [`CONTEXT.md`](../../CONTEXT.md) plays for the universal
DP-SGD language. The narrative pages under
[`mkdocs/algorithms/spiderboost/`](../../mkdocs/algorithms/spiderboost/index.md)
*explain* these terms; they do not redefine them.

Parameter names (`L0`, `L1`, `b1`, `b2`, `T`, `q`) follow Arora et al. 2023. Do not
rename them to be "clearer".

Read [`CONTEXT.md`](../../CONTEXT.md) first for the shared terms (DP, privacy
budget, RDP, per-sample clipping, Poisson subsampling, final/random iterate, …).

---

## Training structure

**Phase**
One complete cycle of one anchor step followed by `q` variation steps. The unit of iteration in Private SpiderBoost.
_Avoid_: epoch, round, cycle

**Phase length (q)**
The number of variation steps per anchor step within one phase. A key hyperparameter: larger `q` amortizes the cost of the anchor step but increases variance accumulation.
_Avoid_: steps per phase, inner loop count, q parameter

> **Note on `q`.** SpiderBoost reuses the symbol `q` for both the phase length and
> the Poisson [sampling rate](../../CONTEXT.md#sampling) — they share notation in
> the paper and are the same hyperparameter in `TrainConfig`. Do not treat them as
> two independent knobs.

**Anchor step**
The step within a phase that computes a full gradient at the current reference point, establishing the variance-reduction baseline. Corresponds to the SPIDER "snapshot" computation.
_Avoid_: snapshot step, reference computation, full-batch step, outer step

**Variation step**
Each of the `q` lightweight steps within a phase that uses a stochastic gradient estimator corrected by the anchor-step baseline. Cheaper per step than an anchor step.
_Avoid_: inner step, correction step, SPIDER step

---

## Noise and clipping

**Three noise scales (σ₁, σ₂, σ₂_hat)**
SpiderBoost injects noise at three points, not one: `σ₁` (anchor step), `σ₂` (variation step), and `σ₂_hat` (bias correction term). All three are calibrated together by `compute_noise_scales` to satisfy the privacy budget. Any change to noise injection must account for all three.
_Avoid_: noise multiplier, sigma, the noise scale (singular)

**Clipping thresholds (L0, L1)**
SpiderBoost uses two [clipping thresholds](../../CONTEXT.md#noise-and-clipping): `L0` bounds per-sample gradients on the anchor step, `L1` bounds them on the variation step. Both follow the paper's notation.
_Avoid_: C, clip norm, max gradient norm

---

## Public API surface

**`train()`**
The entry point for running Private SpiderBoost (`dimma.train`). Takes data, a per-sample loss function, initial params, a config, and noise scales. Returns a `TrainResult`. Future algorithms expose their own entry point under `dimma.algorithms.<name>.train`.

**`TrainConfig`**
A dataclass holding all SpiderBoost hyperparameters: ε, δ, L0, L1, T (total phases), q (phase length / sampling rate), b1 (anchor batch size), b2 (variation batch size), η (learning rate), seed.

**`TrainResult`**
The output of `train()`: contains `params_final` (the [final iterate](../../CONTEXT.md#training-output)), `params_random` (the [random iterate](../../CONTEXT.md#training-output) — the one carrying the formal (ε, δ)-DP guarantee under Algorithm 2), and `history`.

**`compute_noise_scales()`**
Calibrates the three noise scales (σ₁, σ₂, σ₂_hat) from the privacy budget and training config using RDP accounting. Must be called before `train()`.

**`NoiseScales`**
A named triple `(σ₁, σ₂, σ₂_hat)` returned by `compute_noise_scales()`.
