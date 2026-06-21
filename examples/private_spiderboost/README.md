# Private SpiderBoost Examples for dimma

These notebooks reproduce the Private SpiderBoost runs on the Criteo 1M
sample originally implemented in `private_spider_boost_criteo`, adapted
to use the `dimma` library. They serve as a usage demonstration of
`dimma.train` end-to-end and of the privacy-utility tradeoffs surfaced
by Algorithm 2 of Arora et al. (ICML 2023).

## Prerequisites

Install `dimma` (from the library root) and notebook dependencies:

```bash
pip install -e .            # from dimma/
pip install jupyter matplotlib scikit-learn scipy
```

## Layout

```
examples/private_spiderboost/
  notebooks/   # the runnable notebooks (launch Jupyter from here)
  lib/         # shared modules imported by the notebooks: model.py, visualization.py
  scripts/     # standalone helpers (diagnostics, notebook generators)
  figs/        # figures written by the notebooks
```

Each notebook's first cell resolves the example root by walking up from the
current working directory until it finds `lib/model.py`, then puts `lib/` on
`sys.path`. This means the notebooks run correctly whether Jupyter is launched
from `notebooks/` or from the `examples/private_spiderboost/` root.

## Dataset

The notebooks call `dimma.datasets.load_criteo()`, which downloads the
Criteo 1M parquet sample (~30 MB) on first use and caches it under the
OS-appropriate user cache directory. The dataset is licensed
**CC-BY-NC-SA 4.0** (© Criteo Labs). Non-commercial use only;
derivative works must be shared under the same license.

## Notebooks

All notebooks live under `notebooks/`.

| Notebook | View |
|---|---|
| `train_private_spiderboost.ipynb` — reference end-to-end run of Private SpiderBoost on the 13 integer features, with final ROC-AUC and gradient-norm plots. | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Larraguibel/dimma/blob/main/examples/private_spiderboost/notebooks/train_private_spiderboost.ipynb) |
| `phase_length_q_tradeoff.ipynb` — sweep over phase length `q` at fixed `epsilon`. | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Larraguibel/dimma/blob/main/examples/private_spiderboost/notebooks/phase_length_q_tradeoff.ipynb) |
| `privacy_utility_tradeoff.ipynb` — sweep over privacy budget `epsilon` at fixed `q`. | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Larraguibel/dimma/blob/main/examples/private_spiderboost/notebooks/privacy_utility_tradeoff.ipynb) |
| `varying_delta_experiments.ipynb` — sweep over the privacy parameter `delta` at fixed `epsilon`, vs. a non-private SPIDER baseline. | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Larraguibel/dimma/blob/main/examples/private_spiderboost/notebooks/varying_delta_experiments.ipynb) |
| `theorem42_stationarity_rates.ipynb` — empirical verification of Theorem 4.2's α-stationarity rate: sweep `epsilon` with `eta`/`q`/`b2` derived per-`epsilon` from Theorem B.3 via `resolve_config` (`T=2000` keeps the whole sweep in-regime), measure the true empirical gradient norm at the random output step, and fit the bound's unknown constant. | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Larraguibel/dimma/blob/main/examples/private_spiderboost/notebooks/theorem42_stationarity_rates.ipynb) |

## What changed from the original

- **Loss is passed explicitly to `dimma.train`** via `per_sample_loss_fn`,
  rather than being hard-coded inside the training module.
- **Test-set evaluation lives in the notebook**, not the loop. The
  library's training loop does no I/O, no metric computation, no
  evaluation. Notebooks evaluate `params_final` and `params_random`
  after `train` returns.
- **Two independent RNG streams.** `dimma.train` separates the sampling
  RNG (Poisson masks, privacy-relevant) from the control-flow RNG
  (output-step draw, not privacy-relevant). For the same seed, the
  random output step `output_step` will differ from the original; the
  final iterate `params_final` agrees up to floating-point.
- **No more `TrainHistory.train_loss` / `eval_auc`.** The library's
  `TrainHistory` records only `grad_norm`, `wall_time_s`,
  `noise_scales`, and `output_step`. Per-step train-loss tracking
  would require the callback to receive the batch, which the current
  `StepInfo` does not expose — a future library enhancement.
