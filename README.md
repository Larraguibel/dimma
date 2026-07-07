# dimma

JAX-based library of differentially private optimization algorithms.

## Overview

`dimma` is a framework for building differentially private stochastic
optimization algorithms: a set of shared, architecture-agnostic
primitives (per-sample clipping, Gaussian noise, Poisson subsampling)
plus privacy accounting, with each algorithm implemented as a thin
layer on top. The first algorithm implemented is Private SpiderBoost
(Arora et al., ICML 2023), a variance-reduced DP-SGD variant.

Shared foundation (used by every algorithm):

- **Architecture-agnostic training.** An algorithm takes a
  `per_sample_loss_fn` and an `init_params` pytree; the library does
  no model construction, no metric computation, and no I/O. Evaluation
  and bookkeeping live in user code via a per-step callback.
- **Per-sample DP primitives** (`dimma.core`): per-sample clipping,
  Gaussian noise injection, and pytree arithmetic, reused across
  algorithms rather than reimplemented per algorithm.
- **Poisson subsampling** with both the standard rejection-on-oversize
  and a truncated heuristic variant.
- **Privacy accounting** via Google's `dp-accounting` (RDP). Generic
  sampling-based accountants are reusable; algorithm-specific
  accountants live with their algorithm.
- **Dataset loaders** (`dimma.datasets.load_criteo`, ...) with on-disk
  caching, checksum verification, and license attribution.

Implemented algorithms:

- **Private SpiderBoost (Algorithm 2 of Arora et al. 2023)** with
  JIT-compiled anchor / variation step kernels and a
  `compute_noise_scales(epsilon, delta, ...)` call that returns the
  `(sigma1, sigma2, sigma2_hat)` triple the loop needs. Entry point:
  `dimma.train`.

Status: `0.1.0`. The public API may change before `1.0`.

## Documentation

Full project documentation is published at
**<https://larraguibel.github.io/dimma/>** (built with MkDocs, deployed
automatically from `main`). It is the narrative companion to this README
and covers:

- **Differential Privacy for SGD** — the conceptual foundations:
  subsampling, sensitivity, clipping, and privacy accounting.
- **JAX, Flax & Optax tooling** — the primitives used to build
  non-standard DP-SGD variants, and why off-the-shelf libraries don't fit.
- **The dimma library** — module map, design conventions, and how to add
  a new algorithm.
- **Private SpiderBoost** — the first implemented algorithm: the gaps
  between the paper's theory and the code, implementation heuristics,
  and the `q`-invariance of the random-output iterate.

The documentation source lives in [`mkdocs/`](mkdocs/); `docs/` is
reserved for agent-facing context and is not part of the published site.

## Quickstart

The example below trains Private SpiderBoost, the first algorithm in
dimma; `dimma.train` is its entry point. Future algorithms expose their
own entry point under `dimma.algorithms.<name>.train`.

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

See [examples/private_spiderboost/](examples/private_spiderboost/) for runnable notebooks.

## Installation

dimma requires **Python ≥ 3.10**. Installation is two steps: first
install JAX for your hardware, then install dimma with the extras for
what you intend to do.

We recommend a fresh virtual environment:

```bash
git clone https://github.com/Larraguibel/dimma.git
cd dimma
python -m venv .venv && source .venv/bin/activate
```

### Step 1 — Install JAX for your hardware

dimma **never pins JAX** (see [JAX Version](#jax-version)), so you choose
the build that matches your machine. Install it *before* dimma.

```bash
# CPU only (works everywhere)
pip install -U jax

# NVIDIA GPU, CUDA 12 (see GPU setup below)
pip install -U "jax[cuda12]"
```

For TPU, AMD ROCm, or Apple Metal, follow the
[JAX installation guide](https://github.com/google/jax#installation).

### Step 2 — Install dimma for your use case

The three common setups install different extras — pick the row that
matches what you're doing:

| Use case | Command | What you get |
|---|---|---|
| **General use** (import the library in your own code) | `pip install -e .` | the library only |
| **Run the notebooks** under [examples/](examples/) | `pip install -e ".[examples]"` | adds `jupyter`, `matplotlib`, `scikit-learn` |
| **Develop / run the tests** | `pip install -e ".[dev]"` | adds `pytest`, `scikit-learn` |

Extras combine, e.g. to both hack on the library and run the notebooks:

```bash
pip install -e ".[dev,examples]"
```

### Verify the install

```bash
python -c "import jax; print(jax.devices())"
```

This prints the devices JAX will use — `[CpuDevice(id=0)]` on CPU, or
`[CudaDevice(id=0)]` on a working GPU install.

### GPU setup (NVIDIA / CUDA 12)

The `jax[cuda12]` wheels bundle the CUDA and cuDNN libraries, so you only
need a recent **NVIDIA driver** — no system-wide CUDA toolkit. Confirm
the driver is visible first:

```bash
nvidia-smi
```

Then install GPU JAX *before* dimma (Step 1 above), install your extras
(Step 2), and verify:

```bash
python -c "import jax; print(jax.devices())"   # expect [CudaDevice(id=0)]
```

If you instead see `[CpuDevice(id=0)]`, JAX isn't finding the GPU —
almost always a driver/CUDA-version mismatch. Check that `nvidia-smi`
works and that the driver is new enough for CUDA 12. On a cloud box that
already ships GPU JAX, you can skip Step 1, but still run this check.

JAX uses the GPU automatically once detected; no notebook or library
code needs to change.

### Build the documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve   # live preview at http://127.0.0.1:8000
```

## Examples

Runnable Jupyter notebooks live under [examples/private_spiderboost/](examples/private_spiderboost/)
and demonstrate Private SpiderBoost on the Criteo 1M click-prediction
sample (downloaded automatically on first run, ~30 MB). See the
[examples README](examples/private_spiderboost/README.md) for details.

- [`train_private_spiderboost.ipynb`](examples/private_spiderboost/notebooks/train_private_spiderboost.ipynb) —
  end-to-end reference run with final ROC-AUC and gradient-norm plots.
- [`phase_length_q_tradeoff.ipynb`](examples/private_spiderboost/notebooks/phase_length_q_tradeoff.ipynb) —
  sweep over the phase length `q` at fixed `epsilon`.
- [`privacy_utility_tradeoff.ipynb`](examples/private_spiderboost/notebooks/privacy_utility_tradeoff.ipynb) —
  sweep over privacy budget `epsilon` at fixed `q`.
- [`varying_delta_experiments.ipynb`](examples/private_spiderboost/notebooks/varying_delta_experiments.ipynb) —
  sweep over the privacy parameter `delta` at fixed `epsilon`, vs. a
  non-private SPIDER baseline.
- [`theorem42_stationarity_rates.ipynb`](examples/private_spiderboost/notebooks/theorem42_stationarity_rates.ipynb) —
  empirical verification of Theorem 4.2's α-stationarity rate, with
  `eta`/`q`/`b2` derived per-`epsilon` from Theorem B.3 via `resolve_config`.

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
├── models/            # Reference models for testing (MLP, losses)
└── utils/             # Device / misc utilities

tests/                 # Pytest suite (regression vs. reference impls)
examples/private_spiderboost/       # Notebook walkthroughs
mkdocs/                # Published documentation source (-> GitHub Pages)
mkdocs.yml             # MkDocs / Material site configuration
docs/                  # Agent-facing context (not published)
```

### Dependency overview

- **Runtime** (installed by default): `jax`, `flax`, `optax`,
  `dp-accounting`, `pandas`, `pyarrow`, `numpy`.
- **`dev` extras**: `pytest`, `scikit-learn` (used by the regression
  tests against reference implementations).
- **`examples` extras**: `matplotlib`, `scikit-learn`, `jupyter`
  (needed by the notebooks and plotting utilities in
  [examples/private_spiderboost/](examples/private_spiderboost/)).
- **`docs` extras**: `mkdocs`, `mkdocs-material` (needed only to build
  the documentation site in [mkdocs/](mkdocs/) locally).

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

This project uses a `src/` layout. Install the `dev` extras (see
[Installation](#installation)), then run the suite:

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
  url    = {https://github.com/Larraguibel/dimma}
}
```

Update the author list and paper reference once a companion preprint is posted.

## Contributing

Before opening an issue or sending a PR, read:

- [`CONTEXT.md`](CONTEXT.md) — universal DP-SGD glossary; use these terms in code, comments, and issues. Per-algorithm vocabulary lives under [`docs/glossaries/`](docs/glossaries/)
- [`CLAUDE.md`](CLAUDE.md) — build/test commands, conventions, and learned rules for AI agents
- [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md) — how issues are filed and triaged
- [`docs/agents/triage-labels.md`](docs/agents/triage-labels.md) — label vocabulary and when to apply each

Privacy accounting changes and public API changes require human review before implementation (`ready-for-human` label). See the triage guide for details.
