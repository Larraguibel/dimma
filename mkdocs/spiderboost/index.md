# Private Spider-Boost

!!! note

    Algoritmo Private Spider Boost propuesto en Arora et al. (ICML 2023). Esta variante de método del gradiente busca encontrar puntos $\alpha-$ estacionarios con privacidad diferencial. La implementación realizada entrena un *MLP* sobre el dataset de *criteo*, filtrado 1M de filas.

    Artículo completo en [Faster Rates of Convergence to Stationary Points in Differentially Private Optimization](https://proceedings.mlr.press/v202/arora23a/arora23a.pdf).

    Implementación en [https://github.com/Larraguibel/Private_SpiderBoost_Criteo](https://github.com/Larraguibel/Private_SpiderBoost_Criteo)

- [Differences between theory and implementation](theory-vs-implementation.md)
- [Implementation notes (heuristics, deferred decisions)](implementation-notes.md)
- [The q-invariance of params_random](q-invariance.md)

## Implementation in dimma

The algorithm lives in `dimma.algorithms.spiderboost` and is the first algorithm implemented in the project's library. The public entry point is `dimma.train(...)`. See [dimma: the library](../library.md) for the library structure and conventions.

## Sub-pages

- [Differences between theory and implementation](theory-vs-implementation.md)
- [Implementation notes (heuristics, deferred decisions)](implementation-notes.md)
- [The q-invariance of params_random](q-invariance.md)
