# DP-SGD Projects

This section documents a research effort to implement and empirically evaluate differentially private optimization algorithms that go beyond the standard DP-SGD pipeline. It covers the conceptual foundations of DP training (subsampling, sensitivity, accounting), the JAX/Flax/Optax primitives needed to build non-standard variants, and dimma — a small JAX library that codifies those primitives into a model- and dataset-agnostic framework. The first algorithm implemented is Private SpiderBoost, which alternates between anchor and variation steps with two distinct noise scales; its page documents both the implementation and the gaps between the paper and the code. The section is written for people who know DP-SGD theory and are expanding into practice, or who know JAX and are expanding into DP.

- [Differential Privacy for SGD: Overview](overview.md)
- [Jax, Flax and Other DP-SGD Libraries](tooling.md)
- [dimma: the library](library.md)
- [Private Spider-Boost](spiderboost/index.md)
- [Next steps](next-steps.md)

## Structure of this section

!!! note

    A roadmap so you can find the page you need without opening each one.

- **Differential Privacy for SGD: Overview** — conceptual foundations. Why subsampling matters, what an accountant does, the privacy-utility tradeoff.
- **Jax, Flax and Other DP-SGD Libraries** — tooling and patterns. The four JAX primitives, NNX, why we don't use Optax for per-sample work, a minimal end-to-end DP-SGD step.
- **Private Spider-Boost** — first algorithm implemented in dimma.
    - *Differences between theory and implementation* — gap between Theorem 4.2 and what the code actually does.
    - *Implementation notes (heuristics, deferred decisions)* — c=1.0, T+1 vs Theorem B.2, the 2·L₀ sensitivity bound, truncated-Poisson accounting, RNG asymmetry.
    - *The q-invariance of params_random* — algorithmic property worth knowing before interpreting q-sweep plots.
- **dimma: the library** — module map, design conventions, how to add a new algorithm.
- **Next steps** — running TODO list for the project.
