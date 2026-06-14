# Private Spider-Boost

**A differentially private variance-reduced gradient method for finding approximate stationary points in non-convex optimization.**

!!! note

    Algorithm proposed in Arora et al. (ICML 2023). The implementation trains an MLP on the Criteo dataset (filtered to 1M rows).

    Full paper: [Faster Rates of Convergence to Stationary Points in Differentially Private Optimization](https://proceedings.mlr.press/v202/arora23a/arora23a.pdf)

    Original implementation: [https://github.com/Larraguibel/Private_SpiderBoost_Criteo](https://github.com/Larraguibel/Private_SpiderBoost_Criteo)

## Algorithm in brief

Private Spider-Boost alternates between two types of steps:

- **Anchor steps** (every `q` iterations): compute a full or large-batch gradient estimate and re-anchor the variance-reduction term.
- **Variation steps** (all other iterations): compute a small-batch gradient *difference* from the previous iterate, corrected by the last anchor, to keep the variance low.

Each step injects calibrated Gaussian noise. The algorithm uses three distinct noise scales (`σ₁`, `σ₂`, `σ̂₂`) rather than one, because the anchor and variation mechanisms have different sensitivities.

## Implementation in dimma

The algorithm lives in `dimma.algorithms.spiderboost`. The public entry point is `dimma.train(...)`. See [dimma: the library](../../library.md) for the library structure and conventions.

## Pages in this section

- [Differences between theory and implementation](theory-vs-implementation.md)
- [Implementation notes (heuristics, deferred decisions)](implementation-notes.md)
- [The q-invariance of params_random](q-invariance.md)
