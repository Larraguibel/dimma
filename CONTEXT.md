# dimma — Project Context

## What it is

`dimma` is a JAX-based Python library (v0.1.0) implementing differentially private (DP) stochastic optimization algorithms with composable privacy accounting. It is developed as part of the FONDECYT research project.

## Core algorithm

The current release implements **Private SpiderBoost** (Algorithm 2, Arora et al., ICML 2023), a variance-reduced variant of DP-SGD that exploits second-order curvature information for faster convergence under a formal (ε, δ)-DP guarantee. The training loop produces two iterates: the final one and a random intermediate iterate (the one with the formal guarantee).

## Repository layout

```
src/dimma/
├── algorithms/spiderboost/   # train(), TrainConfig, JIT-compiled anchor/variation kernels
├── accounting/               # compute_noise_scales() → (σ1, σ2, σ2_hat); RDP accountants
├── core/
│   ├── sampling/poisson.py   # Poisson subsampling (standard + truncated heuristic)
│   ├── clipping.py           # Per-sample gradient clipping
│   ├── noise.py              # Gaussian noise injection
│   └── pytree.py             # Pytree norm / arithmetic helpers
├── datasets/                 # Cached dataset loaders (Criteo 1M, ...)
└── utils/device.py           # Device helpers

tests/                        # Pytest suite including regression tests vs. reference impls
examples/criteo/              # Three Jupyter notebooks: training, q sweep, ε sweep
```

## Public API (top-level `dimma.*`)

| Symbol | Purpose |
|---|---|
| `train(x, y, per_sample_loss_fn, init_params, config, noise_scales)` | Full training loop |
| `TrainConfig` | Hyperparameters: ε, δ, L0, L1, T, q, b1, b2, η, seed |
| `TrainResult` | `.params_final`, `.params_random`, `.history` |
| `TrainHistory` / `StepInfo` | Per-step callback data |
| `compute_noise_scales(ε, δ, L0, L1, T, q, b1, b2, n)` | Calibrates (σ1, σ2, σ2_hat) via Google dp-accounting (RDP) |
| `NoiseScales` | Named triple returned by `compute_noise_scales` |

The library is **model-agnostic**: it takes a `per_sample_loss_fn` and an `init_params` pytree; all model construction, evaluation, and I/O live in user code.

## Key design choices

- **JAX + JIT kernels**: anchor and variation steps are JIT-compiled for performance.
- **Poisson subsampling**: two variants — standard (rejection on oversize) and truncated heuristic.
- **No pinned JAX version**: users manage their JAX/CUDA environment; dimma declares only a lower bound on Python (`>=3.10`).
- **`src/` layout** with Hatchling build backend.
- **Dataset caching**: Criteo 1M is downloaded on demand with checksum verification; a one-time attribution notice is printed to stderr.

## Dependencies

- **Runtime**: `jax`, `flax`, `optax`, `dp-accounting`, `pandas`, `pyarrow`, `numpy`, `matplotlib`
- **dev extras**: `pytest`, `scikit-learn`
- **examples extras**: `matplotlib`, `scikit-learn`, `jupyter`

## Examples

Three notebooks in `examples/criteo/` demonstrate Private SpiderBoost on Criteo 1M (CC-BY-NC-SA 4.0, ~30 MB, auto-downloaded):

1. `train_private_spiderboost.ipynb` — end-to-end run, ROC-AUC and gradient-norm plots
2. `phase_length_q_tradeoff.ipynb` — sweep over phase length `q` at fixed ε
3. `privacy_utility_tradeoff.ipynb` — sweep over ε at fixed `q`

## Citation

Implements the algorithm from:
> Arora, Bassily, González, Guzmán, Menart, Ullah. *Faster Differentially Private Convex Optimization via Second-Order Methods.* ICML 2023.
