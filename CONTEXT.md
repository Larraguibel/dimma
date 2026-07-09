# dimma — Domain Language

This file is the **universal glossary** for the dimma project: the terms shared by
*every* differentially private optimization algorithm in the library.
One term per concept. Use these names — and only these names — in code, comments,
issues, and prompts.

Terms specific to a single algorithm (its step structure, its noise-scale layout,
its public entry point) do **not** live here. They live in a per-algorithm glossary
under `docs/glossaries/`. See [Algorithm-specific glossaries](#algorithm-specific-glossaries)
at the bottom.

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
An intermediate privacy accounting framework (parameterized by order α) that composes tightly across rounds and converts to (ε, δ)-DP at the end. Used internally by the accounting layer when an algorithm calibrates its noise.
_Avoid_: moment accountant, RDP accounting, Rényi DP

**Privacy accounting**
The process of tracking cumulative privacy loss across training steps. In dimma this is delegated to Google's `dp-accounting` library. Each algorithm injects its own accountant.
_Avoid_: privacy tracking, epsilon tracking

**Post-processing**
Any data-independent transformation applied to an already-private quantity. By the DP post-processing guarantee it consumes NO privacy budget: once a value satisfies (ε, δ)-DP, every function of it that does not re-touch the raw data is (ε, δ)-DP for free. This is what makes the ℓ₁-ball projection privacy-free — it acts only on the already-noised estimate.
_Avoid_: post-hoc processing, output transformation, cleanup step, free step

---

## Noise and clipping

**Noise scale (σ)**
A scalar multiplier on the Gaussian noise added to a gradient, calibrated to satisfy the privacy budget. How many noise scales an algorithm uses, and what each one protects, is algorithm-specific — see the relevant per-algorithm glossary.
_Avoid_: noise multiplier, sigma, standard deviation

**Per-sample gradient clipping**
Bounding each individual sample's gradient to L2-norm ≤ C before aggregation. Required for DP: without it, a single sample could dominate the update.
_Avoid_: gradient clipping, norm clipping, clip

**Clipping threshold (C)**
The maximum L2 norm allowed for a per-sample gradient. An algorithm may use more than one clipping threshold (e.g. a distinct bound per kind of step); the per-algorithm glossary names them.
_Avoid_: clip norm, max gradient norm

**Gradient sparsity (s)**
An upper bound on the number of nonzero coordinates in a per-sample gradient. A required hyperparameter, supplied by the caller and NEVER estimated from data — adapting to an unknown `s` is an open problem, so dimma treats it as a fixed design input, the same way it treats the clipping thresholds. It enters a projection radius through `√s` (a clipped s-sparse vector satisfies `‖·‖₁ ≤ √s · ‖·‖₂`).
_Avoid_: sparsity level, nnz, support size, number of nonzeros

**ℓ₁-ball projection**
Euclidean projection of a noisy gradient estimate onto an ℓ₁-ball — a convex relaxation of the sparse set (every s-sparse, L2-bounded vector lies inside an ℓ₁-ball whose radius scales with `√s`). Applied after calibrated noise as [post-processing](#privacy-guarantees), it converts gradient sparsity into near-dimension-independent error by discarding noise energy in the many directions no sparse signal can occupy. The projected estimate is BIASED.
_Avoid_: l1 projection, sparse projection, l1 clipping, simplex projection

---

## Sampling

**Poisson subsampling**
Selecting each training example independently with probability `q` (the sampling rate), rather than drawing a fixed-size batch. This amplifies the privacy guarantee multiplicatively.
_Avoid_: random sampling, batch sampling, q-sampling

**Sampling rate (q)**
The per-example inclusion probability for Poisson subsampling.
_Avoid_: batch fraction, subsample probability

---

## Training output

**Final iterate**
The model parameters at the end of the last step — the last point in the trajectory.
_Avoid_: last checkpoint, terminal parameters

**Random iterate**
A uniformly sampled intermediate iterate from the training trajectory. In several DP optimization algorithms this — rather than the final iterate — is the point carrying the formal (ε, δ)-DP convergence guarantee. Each algorithm's glossary names how it is returned.
_Avoid_: random checkpoint, sampled params, output with guarantee

---

## Model interface

**Pytree**
A JAX-native nested structure of arrays (dicts, lists, tuples of arrays) used to represent model parameters and gradients. dimma's *algorithms* operate on arbitrary pytrees — they are architecture-agnostic (not tied to any specific network), though the library does ship reference models under `dimma/models/`.
_Avoid_: parameter dict, weight tensors, model weights

**Reference model**
A concrete neural network shipped inside the library (`dimma/models/`, e.g. the Flax `MLP`) so researchers have a testing model in hand. Distinct from the architecture-agnostic algorithms, which never depend on one.
_Avoid_: built-in model, default model, example model

**Per-sample loss function**
A callable `(params, x_i, y_i) → scalar` that computes the loss for a single example. The user provides this; dimma applies `jax.vmap` internally to vectorize over the batch.
_Avoid_: loss function, batch loss, objective

---

## Algorithm-specific glossaries

The terms above are common to all DP-SGD variants. Each algorithm additionally
defines its own vocabulary — step names, noise-scale structure, hyperparameters,
and public entry point — in its own glossary. A per-algorithm glossary is the
canonical naming contract for that algorithm, the same way this file is for the
shared language; the algorithm's pages under `mkdocs/algorithms/` *explain* it,
they do not redefine it.

| Algorithm | Glossary |
|---|---|
| Private SpiderBoost | [`docs/glossaries/spiderboost.md`](docs/glossaries/spiderboost.md) |

When you implement a new algorithm, add a glossary under `docs/glossaries/<name>.md`
and a row to this table. See `docs/agents/domain.md` for the layout rules.
