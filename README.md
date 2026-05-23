# dimma

JAX-based library of differentially private optimization algorithms.

## Overview

`dimma` implements differentially private stochastic optimization
algorithms with composable privacy accounting. The current release
focuses on Private SpiderBoost (Arora et al., ICML 2023), a
variance-reduced DP-SGD variant.

Highlights:

- **Private SpiderBoost (Algorithm 2 of Arora et al. 2023)** with
  JIT-compiled anchor / variation step kernels.
- **Poisson subsampling** with both the standard rejection-on-oversize
  and a truncated heuristic variant.
- **Privacy accounting** via Google's `dp-accounting` (RDP), exposed as
  a single `compute_noise_scales(epsilon, delta, ...)` call that
  returns the `(sigma1, sigma2, sigma2_hat)` triple the loop needs.
- **Model-agnostic training loop.** `train` takes a
  `per_sample_loss_fn` and an `init_params` pytree; the library does
  no model construction, no metric computation, and no I/O. Evaluation
  and bookkeeping live in user code via a per-step callback.
- **Dataset loaders** (`dimma.datasets.load_criteo`, ...) with on-disk
  caching, checksum verification, and license attribution.

Status: `0.1.0`. The public API may change before `1.0`.

## Quickstart

```python
import jax.numpy as jnp
import dimma
from dimma.datasets import load_criteo

# 1. Load data (cached on disk after first download).
x_train, y_train, x_test, y_test = load_criteo()

# 2. Define a per-sample loss. The library privatizes its gradient.
def per_sample_loss(params, x, y):
    logits = x @ params
    return jnp.logaddexp(0.0, logits) - y * logits  # binary logistic loss

init_params = jnp.zeros(x_train.shape[1])

# 3. Configure the run and calibrate noise to the target (epsilon, delta).
config = dimma.TrainConfig(
    epsilon=1.0, delta=1e-6,
    L0=1.0, L1=1.0,
    T=1000, q=50,
    b1=2048, b2=512,
    eta=0.1,
    seed=0,
)
noise_scales = dimma.compute_noise_scales(
    epsilon=config.epsilon, delta=config.delta,
    L0=config.L0, L1=config.L1,
    T=config.T, q=config.q,
    b1=config.b1, b2=config.b2,
    n=x_train.shape[0],
)

# 4. Train. `train` returns both the final iterate and the random-output
#    iterate (the one with the formal DP guarantee under Algorithm 2).
result = dimma.train(
    x_train, y_train,
    per_sample_loss_fn=per_sample_loss,
    init_params=init_params,
    config=config,
    noise_scales=noise_scales,
)

params_dp = result.params_random  # the privately-released model
```

See [examples/criteo/](examples/criteo/) for runnable notebooks.

## Installation

Base install (library only):

```bash
pip install -e .
```

To also run the test suite, install the `dev` extras:

```bash
pip install -e ".[dev]"
```

To run the notebooks under [examples/](examples/) (e.g. the Criteo
experiments), install the `examples` extras:

```bash
pip install -e ".[examples]"
```

You can combine extras, e.g. `pip install -e ".[dev,examples]"`.

## Examples

Runnable Jupyter notebooks live under [examples/criteo/](examples/criteo/)
and demonstrate Private SpiderBoost on the Criteo 1M click-prediction
sample (downloaded automatically on first run, ~30 MB). See the
[examples README](examples/criteo/README.md) for details.

- [`train_private_spiderboost.ipynb`](examples/criteo/train_private_spiderboost.ipynb) —
  end-to-end reference run with final ROC-AUC and gradient-norm plots.
- [`phase_length_q_tradeoff.ipynb`](examples/criteo/phase_length_q_tradeoff.ipynb) —
  sweep over the phase length `q` at fixed `epsilon`.
- [`privacy_utility_tradeoff.ipynb`](examples/criteo/privacy_utility_tradeoff.ipynb) —
  sweep over privacy budget `epsilon` at fixed `q`.

## Project structure

```
src/dimma/
├── algorithms/        # Training loops (currently: spiderboost)
│   └── spiderboost/   #   - train(), TrainConfig, step kernels
├── accounting/        # Privacy accounting
│                      #   - compute_noise_scales(), NoiseScales
│                      #   - sampling-based RDP accountants
├── core/              # Algorithm-agnostic primitives
│   ├── sampling/      #   - Poisson subsampling (standard / truncated)
│   ├── clipping.py    #   - per-sample gradient clipping
│   ├── noise.py       #   - Gaussian noise injection
│   └── pytree.py      #   - pytree norm / arithmetic helpers
├── datasets/          # Cached dataset loaders (Criteo, ...)
└── utils/             # Device / misc utilities

tests/                 # Pytest suite (regression vs. reference impls)
examples/criteo/       # Notebook walkthroughs
```

### Dependency overview

- **Runtime** (installed by default): `jax`, `flax`, `optax`,
  `dp-accounting`, `pandas`, `pyarrow`, `numpy`.
- **`dev` extras**: `pytest`, `scikit-learn` (used by the regression
  tests against reference implementations).
- **`examples` extras**: `matplotlib`, `scikit-learn`, `jupyter`
  (needed by the notebooks and plotting utilities in
  [examples/criteo/](examples/criteo/)).

## Datasets and licensing

`dimma.datasets` provides convenience loaders for canonical benchmark
datasets. Datasets are downloaded on demand to a user-controlled cache
directory (see ``dimma.datasets._cache.get_cache_dir``).

Each dataset retains its original license:

- **Criteo 1M** (`dimma.datasets.load_criteo`): CC-BY-NC-SA 4.0. Original
  data © Criteo Labs. Non-commercial use only. Derivative works must
  be shared under the same license.

The library itself does not change these licenses. Users of `dimma` are
responsible for complying with the license of each dataset they load.
The library prints a one-time attribution notice to stderr on first
download per process.

## JAX Version

JAX version is **not pinned** by this library. Manage JAX (and its CUDA variant if
applicable) in your own environment before installing dimma. See the
[JAX installation guide](https://github.com/google/jax#installation).

## Development

This project uses a `src/` layout. After cloning, install in editable mode:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Citation

If you use `dimma` in academic work, please cite the algorithm paper
and (until a `dimma` preprint is available) this repository:

```bibtex
@inproceedings{arora2023spiderboost,
  title     = {Faster Differentially Private Convex Optimization via
               Second-Order Methods},
  author    = {Arora, Raman and Bassily, Raef and Gonz{\'a}lez, Tom{\'a}s
               and Guzm{\'a}n, Crist{\'o}bal and Menart, Michael and
               Ullah, Enayat},
  booktitle = {Proceedings of the 40th International Conference on
               Machine Learning (ICML)},
  year      = {2023}
}

@software{dimma,
  title  = {dimma: JAX-based differentially private optimization
            algorithms},
  author = {Larraguibel, Javier},
  year   = {2026},
  url    = {https://github.com/<your-username>/dimma}
}
```

Replace `<your-username>` with the GitHub user/organization the repo
is hosted under, and update the author list / paper reference once a
companion preprint is posted.
