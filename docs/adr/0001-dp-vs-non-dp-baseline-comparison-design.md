# ADR-0001: DP vs. non-DP baseline comparison design in example notebooks

**Status:** Accepted

## Context

The example notebooks in `examples/private_spiderboost/notebooks/` demonstrate Private SpiderBoost on Criteo. A natural question for readers is: how much does the privacy mechanism cost in terms of model quality? To answer this, three notebooks (`train_private_spiderboost`, `privacy_utility_tradeoff`, `varying_delta_experiments`) include a non-private baseline run alongside the DP run.

This comparison is non-trivial because Private SpiderBoost is step-based (parameterized by T and q), while a reader might expect a baseline expressed in epochs. Several design choices must be made consistently.

## Decisions

### 1. Gradient budget unit: expected gradient evaluations

The comparison x-axis and the budget-matching criterion use **expected gradient evaluations**, defined as:

```
expected_grad_evals(T, q, b1, b2) = (T // q + 1) * b1 + (T - T // q) * 2 * b2
```

Where `T // q + 1` is the number of anchor steps (each evaluating b1 per-sample gradients once) and `T - T // q` is the number of variation steps (each evaluating b2 per-sample gradients twice — once at `params_t`, once at `params_prev`).

**Why expected, not realized:** The DP run uses Poisson subsampling, so realized batch sizes are random. The algorithm's convergence and privacy guarantees are both stated in terms of expected quantities (sampling rate p = b/n). Using realized counts would make the x-axis stochastic and inconsistent with the algorithm's own analysis.

**Why not epochs:** Epochs are not a meaningful unit here. The anchor/variation cost asymmetry (b1 ≫ b2, and variation steps cost 2×b2) means a step-count or epoch-count would misrepresent the actual computational budget.

**Why not wall time:** Wall time conflates hardware and JIT warm-up with algorithmic cost. Expected gradient evals are hardware-independent and reproducible.

### 2. Non-DP baseline: non-private SPIDER with only eta free

The non-DP baseline uses `train_spider` from `lib/model.py` — the non-private SPIDER algorithm with the same anchor/variation step structure as Private SpiderBoost.

**Fixed to match the DP run:** T, q, b1, b2. These four parameters fully determine the gradient budget via the formula above. Changing any of them would change the budget and break the comparison.

**Free to be tuned:** eta (learning rate) only. In the DP run, eta is tuned under heavily noisy gradients (noise scales σ₁, σ₂ are large relative to the signal). Without noise, a different — typically larger — eta is optimal. Forcing the non-DP baseline to use the DP eta would penalize it unfairly.

**Why not free b1/b2:** Freeing b1 or b2 changes the expected gradient eval count, which breaks the budget equivalence. The non-DP baseline must consume the same budget as the DP run.

**Why non-private SPIDER, not SGD/Adam:** Using the same algorithmic structure (anchor + variation steps) isolates the cost of the privacy mechanism. Comparing against SGD/Adam would conflate "cost of DP" with "cost of variance reduction."

**eta sweep:** A visible sweep cell runs `eta ∈ {0.001, 0.01, 0.1}` and selects the best eta. Best is defined in decision 3.

### 3. Eta selection criterion: params_random AUC

The best eta for the non-DP baseline is selected by test AUC evaluated on `params_random` (the uniformly sampled random iterate), not `params_final`.

**Why params_random:** The DP run's output rule returns `params_random` — a uniformly sampled iterate from the training trajectory. This is the iterate with the formal convergence guarantee. Comparing `params_random` to `params_random` controls for the output rule; comparing DP `params_random` to non-DP `params_final` would give the non-DP baseline an advantage unrelated to the privacy cost.

## Consequences

- The three-notebook comparison is internally consistent: same budget formula, same output rule, same baseline algorithm.
- The non-DP baseline's eta will differ from the DP run's eta. This is intentional and should be disclosed in the notebook prose.
- Adding a new notebook with a DP baseline should follow this same design.
- If `train_spider`'s loop structure ever diverges from `dimma.train`'s loop structure (e.g., different indexing convention), the budget formula would no longer be symmetric and this ADR would need revisiting.
