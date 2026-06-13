"""Generator for ../notebooks/theorem42_stationarity_rates.ipynb.

Run from anywhere: ``python examples/private_spiderboost/scripts/build_theorem42_notebook.py``.
Keeps the notebook source reviewable as plain Python and produces clean JSON.

Convention: markdown blocks are wrapped in triple double-quotes; code blocks in
triple single-quotes. Embedded code uses ``#`` comments instead of triple-quote
docstrings so the wrapper delimiters never collide with the cell content.
"""

from __future__ import annotations

import json
from pathlib import Path

cells = []


def _cell_id() -> str:
    return f"cell-{len(cells):02d}"


def md(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------------------
md(r"""
# Empirical verification of Theorem 4.2 (α-stationarity rates) on Criteo

Private SpiderBoost (Algorithm 2 of Arora et al. 2023) carries a theoretical
guarantee — **Theorem 4.2** — bounding the true gradient norm of the empirical
risk at the random output step $w_{t^*}$:

$$\;\|\nabla F(w_{t^*}; S)\| \;=\; O\!\left(\left(\frac{\sqrt{F_0\,L_1\,L_0\,\sqrt{d}\,\log(1/\delta)}}{n\,\varepsilon}\right)^{2/3} \;+\; \frac{L_0\,\sqrt{d}\,\log(1/\delta)}{n\,\varepsilon}\right).$$

The $O(\cdot)$ suppresses an unknown universal constant $C$. This notebook:

1. Sweeps the privacy budget $\varepsilon$, running Private SpiderBoost at each value.
2. Computes the **true empirical gradient norm** at the random output step —
   the quantity the theorem actually bounds, *not* the noisy running estimate
   logged during training.
3. Fits the unknown constant $C$ by comparing observed norms to the theoretical
   rate via `scipy.optimize.curve_fit`.
4. Produces log-log plots of observed norms vs. the fitted theoretical curve.

### What this notebook can and cannot conclude

- It **cannot** recover the true universal constant from the proof; it fits an
  empirical $C$ for *this* Criteo instantiation. A large $C \gg 1$ is the
  scientifically expected outcome (see the closing **Limitations** section) and
  does **not** falsify the theorem.
- It uses **default hyperparameters** (`b1 << n`, `q` not derived from the
  proof). The paper's tightest rate uses `b1 = n`; here `b1 = 8192`.
- It covers only the **empirical** risk (Theorem 4.2), not population risk
  (Theorem 4.3).

> **Key conceptual distinction.** `TrainResult.history.grad_norm[t]` is the L2
> norm of the *noisy* Algorithm 2 gradient **estimate** at step `t` — it
> includes injected Gaussian noise and can be orders of magnitude larger than
> the true gradient. Theorem 4.2 bounds $\|\nabla F(w_{t^*}; S)\|$, the true
> gradient of the mean empirical loss at the random output step. These are
> different quantities; §8c visualizes the gap.
""")

# ---------------------------------------------------------------------------
md("## 1. Imports")

code(r'''
import sys
from pathlib import Path

# Resolve the example root (contains lib/ and figs/) so this runs from anywhere.
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'lib' / 'model.py').is_file())
if str(ROOT / 'lib') not in sys.path:
    sys.path.insert(0, str(ROOT / 'lib'))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import dimma
from dimma import TrainConfig, compute_noise_scales
from dimma.datasets import load_criteo
from dimma.core.pytree import (
    pytree_add,
    pytree_scale,
    pytree_zeros_like,
    pytree_global_norm,
)

import model
''')

# ---------------------------------------------------------------------------
md("""
## 2. Data and sweep configuration

The cleanest axis for verifying Theorem 4.2 is $\\varepsilon$, because the
rate's $\\varepsilon$-dependence is explicit: term 1 $\\propto \\varepsilon^{-2/3}$,
term 2 $\\propto \\varepsilon^{-1}$. All other hyperparameters are held constant.

**On `d`:** Theorem 4.2's `d` is the optimization problem dimension, i.e. the
number of **model parameters** ($\\approx 3{,}201$ for `HIDDEN_DIMS=(64,32)` on
13 input features), **not** the feature dimension 13. We compute it as
`sum(leaf.size for leaf in jax.tree_util.tree_leaves(init_params))`.
""")

code(r'''
FIGS_DIR = ROOT / 'figs'
FIGS_DIR.mkdir(exist_ok=True)

data = load_criteo(features='integer', test_fraction=0.2, seed=0, device='cpu')
x_train, y_train = data.x_train, data.y_train
n_train, d_features = x_train.shape

# --- Sweep design -----------------------------------------------------------
EPSILONS = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
N_SEEDS = 3
SEEDS = list(range(N_SEEDS))

HIDDEN_DIMS = (64, 32)

# Held-constant algorithm/privacy hyperparameters.
L0 = 3.0
L1 = 5.0
BASE = dict(L0=L0, L1=L1, T=200, q=30, b1=8192, b2=512, eta=0.01)

# delta = 1/n is the standard "less than one over dataset size" choice; it
# feeds both compute_noise_scales and the log(1/delta) factor in the bound.
DELTA = 1.0 / n_train

# d = parameter-space dimension.
_probe_params = model.init_params(
    jax.random.PRNGKey(0), input_dim=d_features, hidden_dims=HIDDEN_DIMS
)
d_params = int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(_probe_params)))

print(f'n_train={n_train}  d_features={d_features}  d_params={d_params}')
print(f'delta = 1/n = {DELTA:.3e}')
print(f'EPSILONS = {EPSILONS}   N_SEEDS = {N_SEEDS}')
''')

# ---------------------------------------------------------------------------
md(r"""
## 3. Utility functions

### `compute_true_grad_norm` — the quantity the theorem bounds

`jax.grad(model.batch_bce_loss)(params, x, y)` returns the **mean** gradient
over the batch. Running it on all ~800k samples at once risks OOM, so we
accumulate in chunks: for a chunk of size $m$, the chunk mean gradient
weighted by $m/n$ contributes $\frac{1}{n}\sum_{\text{chunk}} \nabla\ell_i$, and
summing across chunks yields the exact full-dataset mean gradient.

### `theorem_bound_terms` — the two terms of the rate

$$\text{term1}(\varepsilon)=\left(\frac{\sqrt{F_0 L_1 L_0 \sqrt{d}\log(1/\delta)}}{n\varepsilon}\right)^{2/3},\qquad
\text{term2}(\varepsilon)=\frac{L_0 \sqrt{d}\log(1/\delta)}{n\varepsilon}.$$
""")

code(r'''
@jax.jit
def _chunk_grad(params, x_chunk, y_chunk):
    return jax.grad(model.batch_bce_loss)(params, x_chunk, y_chunk)


def compute_true_grad_norm(params, x, y, chunk_size=8192):
    # L2 norm of the true mean-empirical-loss gradient ||nabla F(params; S)||.
    # Deterministic: no noise, no subsampling. Accumulates chunk mean gradients
    # weighted by (chunk_size / n_total), so the result equals a single
    # jax.grad(batch_bce_loss) over the whole set, computed without OOM.
    n_total = int(x.shape[0])
    acc = pytree_zeros_like(params)
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        g = _chunk_grad(params, jnp.asarray(x[start:end]), jnp.asarray(y[start:end]))
        acc = pytree_add(acc, pytree_scale(g, (end - start) / n_total))
    return float(pytree_global_norm(acc))


def compute_mean_loss(params, x, y, chunk_size=8192):
    # Mean BCE loss F(params; S) over the full set, chunked to avoid OOM.
    n_total = int(x.shape[0])
    total = 0.0
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_loss = float(
            model.batch_bce_loss(params, jnp.asarray(x[start:end]), jnp.asarray(y[start:end]))
        )
        total += chunk_loss * (end - start) / n_total
    return total


def theorem_bound_terms(eps, F0, L0=L0, L1=L1, d=d_params, delta=DELTA, n=n_train):
    # Return (term1, term2) of the Theorem 4.2 rate at privacy budget eps.
    # term1 ~ eps^{-2/3}, term2 ~ eps^{-1}. Both exclude the unknown constant C.
    log_inv_delta = np.log(1.0 / delta)
    sqrt_d = np.sqrt(d)
    inner = np.sqrt(F0 * L1 * L0 * sqrt_d * log_inv_delta) / (n * eps)
    term1 = inner ** (2.0 / 3.0)
    term2 = L0 * sqrt_d * log_inv_delta / (n * eps)
    return term1, term2
''')

# ---------------------------------------------------------------------------
md("""
### 3a. Spot-checks for the utility functions

Inline assertions (in the spirit of `varying_delta_experiments.ipynb`) that
catch bugs before any expensive sweeping:

- **Determinism:** `compute_true_grad_norm` on the same params twice is identical.
- **Sensitivity:** two different param pytrees give different norms.
- **Chunked vs. full-dataset consistency:** on a 1,000-sample subset, chunked
  accumulation (`chunk_size=100`) matches a direct full call to within 1e-5
  relative error.
""")

code(r'''
_k = jax.random.PRNGKey(123)
_pA = model.init_params(_k, input_dim=d_features, hidden_dims=HIDDEN_DIMS)
_pB = model.init_params(jax.random.PRNGKey(456), input_dim=d_features, hidden_dims=HIDDEN_DIMS)

# Determinism: same params -> identical norm (no randomness in the computation).
_n1 = compute_true_grad_norm(_pA, x_train[:2000], y_train[:2000], chunk_size=512)
_n2 = compute_true_grad_norm(_pA, x_train[:2000], y_train[:2000], chunk_size=512)
assert _n1 == _n2, f'non-deterministic: {_n1} != {_n2}'

# Sensitivity: different params -> different norm.
_nB = compute_true_grad_norm(_pB, x_train[:2000], y_train[:2000], chunk_size=512)
assert abs(_n1 - _nB) > 1e-8, 'two different param trees gave the same grad norm'

# Chunked vs. full-dataset consistency on a 1,000-sample subset.
_xs, _ys = x_train[:1000], y_train[:1000]
_chunked = compute_true_grad_norm(_pA, _xs, _ys, chunk_size=100)
_full_g = jax.grad(model.batch_bce_loss)(_pA, jnp.asarray(_xs), jnp.asarray(_ys))
_full = float(pytree_global_norm(_full_g))
_rel = abs(_chunked - _full) / _full
assert _rel < 1e-5, f'chunked vs full mismatch: rel error {_rel:.2e}'

print(f'determinism OK   sensitivity OK   chunked-vs-full rel error = {_rel:.2e}')
''')

# ---------------------------------------------------------------------------
md("""
## 4. Initial loss $F_0$ from the actual random initialization

The bound's $F_0$ is the initial suboptimality $F(w_0;S) - F^*(S)$. We report
both:

- `F0_upper = F(w_0; S)` — an upper bound (assumes $F^* = 0$),
- `F0_approx = F(w_0; S) - 0.4` — subtracting an approximate Criteo minimum BCE.

The theoretical curve is shown under both assumptions. $F_0$ is averaged over
the random initializations actually used in the sweep.
""")

code(r'''
F0_per_seed = []
for seed in SEEDS:
    init_p = model.init_params(
        jax.random.PRNGKey(seed), input_dim=d_features, hidden_dims=HIDDEN_DIMS
    )
    F0_per_seed.append(compute_mean_loss(init_p, x_train, y_train))

F0_upper = float(np.mean(F0_per_seed))      # F(w0; S), assumes F* = 0
F0_approx = max(F0_upper - 0.4, 1e-6)        # subtract approx Criteo min BCE
print(f'F0_upper  = F(w0;S)        = {F0_upper:.4f}')
print(f'F0_approx = F(w0;S) - 0.4  = {F0_approx:.4f}')
print(f'(per-seed F(w0;S): {[round(v, 4) for v in F0_per_seed]})')
''')

# ---------------------------------------------------------------------------
md("""
## 5. Non-private SPIDER baseline

`model.train_spider` runs non-private SPIDER with the **same** anchor/variation
structure and the **same** random-output-step rule as the DP loop, so its
`params_random` is directly comparable. This quantifies the cost of differential
privacy on stationarity.
""")

code(r'''
print('Training non-private SPIDER baseline...')
baseline_norms = []
for seed in SEEDS:
    init_p = model.init_params(
        jax.random.PRNGKey(seed), input_dim=d_features, hidden_dims=HIDDEN_DIMS
    )
    res_np = model.train_spider(
        x_train, y_train, init_params=init_p,
        T=BASE['T'], q=BASE['q'], b1=BASE['b1'], b2=BASE['b2'],
        eta=BASE['eta'], seed=seed,
    )
    gn = compute_true_grad_norm(res_np.params_random, x_train, y_train)
    baseline_norms.append(gn)
    print(f'  seed={seed}  t*={res_np.history.output_step:3d}  '
          f'||nabla F(w*)||={gn:.4e}  ({sum(res_np.history.wall_time_s):.1f}s)')

baseline_mean = float(np.mean(baseline_norms))
print(f'Non-private baseline true grad norm (mean over {N_SEEDS} seeds): {baseline_mean:.4e}')
''')

# ---------------------------------------------------------------------------
md("""
## 6. The ε sweep

For each $\\varepsilon$ and each seed: run Private SpiderBoost, then compute the
**true** empirical gradient norm at `result.params_random` (the uniformly random
output step $t^* \\sim \\text{Uniform}\\{1,\\dots,T\\}$ — Theorem 4.2's output
rule). We deliberately use `params_random`, **not** `params_final`.

We also retain, for one representative run ($\\varepsilon=1$, seed 0), the full
**noisy** running-estimate trajectory `history.grad_norm` so §8c can contrast it
with the true norm.
""")

code(r'''
# norms[eps] -> list of true grad norms (one per seed)
norms = {eps: [] for eps in EPSILONS}
output_steps = {eps: [] for eps in EPSILONS}
single_run = {}  # representative run for the noisy-vs-true visualization

for eps in EPSILONS:
    print(f'--- eps = {eps} ---')
    for seed in SEEDS:
        cfg = TrainConfig(epsilon=eps, delta=DELTA, seed=seed, **BASE)
        noise_scales = compute_noise_scales(
            L0=cfg.L0, L1=cfg.L1, epsilon=cfg.epsilon, delta=cfg.delta,
            T=cfg.T, q=cfg.q, n=n_train, b1=cfg.b1, b2=cfg.b2,
        )
        init_p = model.init_params(
            jax.random.PRNGKey(seed), input_dim=d_features, hidden_dims=HIDDEN_DIMS
        )
        res = dimma.train(
            x_train, y_train,
            per_sample_loss_fn=model.per_sample_bce_loss,
            init_params=init_p, config=cfg, noise_scales=noise_scales,
            sampler='poisson',
        )
        gn = compute_true_grad_norm(res.params_random, x_train, y_train)
        norms[eps].append(gn)
        output_steps[eps].append(res.history.output_step)
        if eps == 1.0 and seed == 0:
            single_run['eps1_seed0'] = dict(
                noisy=list(res.history.grad_norm),
                output_step=res.history.output_step,
                true_norm=gn,
            )
        print(f'  seed={seed}  t*={res.history.output_step:3d}  '
              f'||nabla F(w*)||={gn:.4e}')

mean_norms = np.array([float(np.mean(norms[e])) for e in EPSILONS])
sem_norms = np.array([
    float(np.std(norms[e], ddof=1) / np.sqrt(N_SEEDS)) if N_SEEDS > 1 else 0.0
    for e in EPSILONS
])
eps_arr = np.array(EPSILONS, dtype=float)

print('\nSummary (true gradient norm at w_t*):')
for e, m, s in zip(EPSILONS, mean_norms, sem_norms):
    print(f'  eps={e:5.1f}  mean={m:.4e}  sem={s:.2e}')
''')

# ---------------------------------------------------------------------------
md("""
### 6a. Monotonicity assertion

A fundamental property: more privacy budget (larger $\\varepsilon$) means less
noise, so the mean true gradient norm should **decrease** as $\\varepsilon$
increases. A violation indicates a bug, not merely a loose bound. We allow a
small tolerance for the variance of the random output step $t^*$.
""")

code(r'''
_tol = 0.10  # relative tolerance for t*-induced variance between adjacent points
_violations = []
for i in range(len(EPSILONS) - 1):
    lo, hi = mean_norms[i], mean_norms[i + 1]
    if hi > lo * (1 + _tol):
        _violations.append((EPSILONS[i], EPSILONS[i + 1], lo, hi))

if _violations:
    for e0, e1, lo, hi in _violations:
        print(f'  VIOLATION: norm(eps={e1}) = {hi:.3e} > norm(eps={e0}) = {lo:.3e}')
assert not _violations, (
    'Monotonicity violated beyond tolerance: gradient norm increased with eps. '
    'This signals a bug (e.g. wrong output step, noise miscalibration), not a loose bound.'
)
print(f'Monotonicity OK (mean true grad norm non-increasing in eps within {_tol:.0%} tol)')
''')

# ---------------------------------------------------------------------------
md(r"""
## 7. Fitting the constant $C$

### 7a. Single-constant fit (primary result)

Fit $C$ in $\widehat{\alpha}(\varepsilon) = C\,[\text{term1}(\varepsilon) + \text{term2}(\varepsilon)]$
via `scipy.optimize.curve_fit`, weighting by the per-point SEM. This is the
headline number: the fitted constant and its uncertainty.
""")

code(r'''
t1_arr, t2_arr = theorem_bound_terms(eps_arr, F0_upper)
bound_full = t1_arr + t2_arr  # C = 1 reference (upper-F0)


def _single_const_model(eps, C):
    t1, t2 = theorem_bound_terms(eps, F0_upper)
    return C * (t1 + t2)


_sigma = sem_norms if (N_SEEDS > 1 and np.all(sem_norms > 0)) else None
popt, pcov = curve_fit(
    _single_const_model, eps_arr, mean_norms,
    p0=[1.0], bounds=([0.0], [np.inf]), sigma=_sigma, absolute_sigma=False,
)
C_fit = float(popt[0])
C_err = float(np.sqrt(pcov[0, 0]))
print(f'Single-constant fit:  C = {C_fit:.4g}  +/-  {C_err:.2g}')
print('(C >> 1 is expected and valid; see the Limitations section.)')
''')

md(r"""
### 7b. Two-constant diagnostic fit

Fit $C_1, C_2$ separately (one per term):
$\widehat{\alpha}(\varepsilon) = C_1\,\text{term1}(\varepsilon) + C_2\,\text{term2}(\varepsilon)$.
If $C_1 \approx C_2$, the single-constant model is appropriate. If they differ
substantially, the two terms carry different empirical weights.
""")

code(r'''
def _two_const_model(eps, C1, C2):
    t1, t2 = theorem_bound_terms(eps, F0_upper)
    return C1 * t1 + C2 * t2

try:
    popt2, pcov2 = curve_fit(
        _two_const_model, eps_arr, mean_norms,
        p0=[1.0, 1.0], bounds=([0.0, 0.0], [np.inf, np.inf]),
        sigma=_sigma, absolute_sigma=False, maxfev=10000,
    )
    C1_fit, C2_fit = float(popt2[0]), float(popt2[1])
    C1_err, C2_err = float(np.sqrt(pcov2[0, 0])), float(np.sqrt(pcov2[1, 1]))
    print(f'Two-constant fit:  C1 (term1, eps^-2/3) = {C1_fit:.4g} +/- {C1_err:.2g}')
    print(f'                   C2 (term2, eps^-1)   = {C2_fit:.4g} +/- {C2_err:.2g}')
    _ratio = C1_fit / C2_fit if C2_fit > 0 else float('inf')
    print(f'                   C1/C2 = {_ratio:.3g}  '
          f'({"comparable" if 0.33 < _ratio < 3 else "substantially different"})')
except RuntimeError as exc:
    C1_fit = C2_fit = None
    print(f'Two-constant fit did not converge: {exc}')
''')

md(r"""
### 7c. Model-free log-linear slope check

Fit $\log(\text{mean\_norm}) = \text{slope}\cdot\log(\varepsilon) + \text{intercept}$
via `numpy.polyfit`. The theoretical prediction lies between $-2/3$ (term 1
dominates) and $-1$ (term 2 dominates). For the Criteo regime the two terms are
comparable near $\varepsilon=1$, so the expected slope is roughly $-0.75$ to
$-0.85$.
""")

code(r'''
log_eps = np.log(eps_arr)
log_norm = np.log(mean_norms)
slope, intercept = np.polyfit(log_eps, log_norm, 1)
print(f'Empirical log-log slope:  {slope:.3f}')
print('Theoretical range:        [-1.0 (term2)  ..  -0.667 (term1)]')
print('Expected (comparable terms near eps=1): ~ -0.75 to -0.85')
''')

md(r"""
### 7d. "Bound is an upper bound" check

Theorem 4.2 bounds the **expectation**, so individual runs may exceed
$C_{\text{fit}}\cdot\text{bound}(\varepsilon)$, but the per-$\varepsilon$ **mean**
should not. We assert the mean lies below the fitted bound at every
$\varepsilon$ and report the fraction of individual runs that do too.
""")

code(r'''
fitted_bound = C_fit * bound_full
below_mean = mean_norms <= fitted_bound * (1 + 1e-6)
indiv_below = []
for i, e in enumerate(EPSILONS):
    frac = np.mean([v <= fitted_bound[i] for v in norms[e]])
    indiv_below.append(frac)
    print(f'  eps={e:5.1f}  mean<=bound: {bool(below_mean[i])}   '
          f'individual runs below bound: {frac:.0%}')

assert np.all(below_mean), (
    'Fitted bound does not upper-bound the per-eps mean at every point. '
    'For a least-squares fit a tiny overshoot is possible; investigate if large.'
)
print('Upper-bound check OK (fitted C bound >= mean at every eps).')
''')

# ---------------------------------------------------------------------------
md("""
## 8. Figures

Saved to `examples/private_spiderboost/figs/` with consistent `theorem42_*` naming so they
can be dropped into a paper without re-running the notebook.
""")

md("### 8a. Main ε-sweep: observed norms vs. fitted theoretical curve (log-log)")

code(r'''
eps_dense = np.logspace(np.log10(min(EPSILONS)), np.log10(max(EPSILONS)), 200)
t1_dense, t2_dense = theorem_bound_terms(eps_dense, F0_upper)
t1_dense_a, t2_dense_a = theorem_bound_terms(eps_dense, F0_approx)

fig, ax = plt.subplots(figsize=(8, 6))

# Observed true gradient norms with SEM error bars over seeds.
ax.errorbar(
    eps_arr, mean_norms, yerr=sem_norms, fmt='o', ms=7, capsize=4,
    color='C0', label='observed $\\|\\nabla F(w_{t^*};S)\\|$ (mean ± SEM)', zorder=5,
)
# Individual seed points to show t*-variance.
for e in EPSILONS:
    ax.scatter([e] * len(norms[e]), norms[e], s=14, color='C0',
               alpha=0.25, zorder=4)

# Fitted single-constant curve and its decomposition (upper-F0).
ax.plot(eps_dense, C_fit * (t1_dense + t2_dense), '-', color='C3', lw=2,
        label=f'fitted $C\\cdot$(term1+term2), $C={C_fit:.3g}$')
ax.plot(eps_dense, C_fit * t1_dense, '--', color='C1', lw=1.5,
        label='$C\\cdot$term1 ($\\propto\\varepsilon^{-2/3}$)')
ax.plot(eps_dense, C_fit * t2_dense, ':', color='C2', lw=1.5,
        label='$C\\cdot$term2 ($\\propto\\varepsilon^{-1}$)')

# Theoretical shape under the F0_approx assumption (scaled by the same C_fit).
ax.plot(eps_dense, C_fit * (t1_dense_a + t2_dense_a), '-', color='0.6', lw=1,
        alpha=0.8, label='fitted curve under $F_0 - 0.4$')

# Non-private baseline.
ax.axhline(baseline_mean, color='k', ls='-.', lw=1.2,
           label=f'non-private SPIDER ({baseline_mean:.2e})')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('privacy budget $\\varepsilon$')
ax.set_ylabel('true empirical gradient norm $\\|\\nabla F(w_{t^*};S)\\|$')
ax.set_title('Theorem 4.2 stationarity rate on Criteo\n'
             f'log-log slope = {slope:.2f} (theory: $-2/3$ to $-1$)')
ax.legend(fontsize=8, loc='best')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / 'theorem42_epsilon_sweep.png', dpi=150)
print('saved', FIGS_DIR / 'theorem42_epsilon_sweep.png')
''')

md("### 8b. Relative residuals between observed and fitted values")

code(r'''
fitted_at_eps = C_fit * bound_full
rel_resid = (mean_norms - fitted_at_eps) / fitted_at_eps

fig, ax = plt.subplots(figsize=(8, 4))
ax.axhline(0, color='k', lw=0.8)
ax.errorbar(eps_arr, rel_resid, yerr=sem_norms / fitted_at_eps,
            fmt='o', capsize=4, color='C3')
ax.set_xscale('log')
ax.set_xlabel('privacy budget $\\varepsilon$')
ax.set_ylabel('(observed - fitted) / fitted')
ax.set_title('Relative residuals of the single-constant fit')
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / 'theorem42_residuals.png', dpi=150)
print('saved', FIGS_DIR / 'theorem42_residuals.png')
''')

md(r"""
### 8c. Noisy running estimate vs. true gradient norm (single run)

This makes the **key conceptual distinction** visceral. The blue trajectory is
`history.grad_norm` — the *noisy* Algorithm 2 estimate at each step, dominated
by injected Gaussian noise. The red marker is the *true* gradient norm
$\|\nabla F(w_{t^*};S)\|$ at the random output step — the quantity Theorem 4.2
actually bounds. They differ by orders of magnitude.
""")

code(r'''
sr = single_run['eps1_seed0']
noisy = np.array(sr['noisy'])
t_star = sr['output_step']

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.arange(len(noisy)), noisy, color='C0', lw=1,
        label='noisy estimate $\\|\\hat\\nabla_t\\|$ (history.grad_norm)')
ax.scatter([t_star], [sr['true_norm']], color='C3', s=90, zorder=5,
           label=f'true $\\|\\nabla F(w_{{t^*}};S)\\|$ at $t^*={t_star}$')
ax.axvline(t_star, color='C3', ls=':', lw=1, alpha=0.7)
ax.set_yscale('log')
ax.set_xlabel('SpiderBoost step $t$')
ax.set_ylabel('gradient norm (log scale)')
ax.set_title('Noisy running estimate vs. true gradient norm\n'
             '($\\varepsilon=1$, seed 0) - the theorem bounds the red point, not the blue curve')
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / 'theorem42_noisy_vs_true.png', dpi=150)
print('saved', FIGS_DIR / 'theorem42_noisy_vs_true.png')
print(f'true norm at t*: {sr["true_norm"]:.4e}   '
      f'noisy estimate at t*: {noisy[t_star]:.4e}   '
      f'ratio: {noisy[t_star] / sr["true_norm"]:.1f}x')
''')

# ---------------------------------------------------------------------------
md(r"""
## 9. Limitations — what this notebook cannot conclude

This comparison is **diagnostic**, not a proof check. Specifically:

1. **The constant $C$ is unknown and fitted, not derived.** $C \gg 1$ is the
   scientifically valid outcome. The proof's $O(\cdot)$ suppresses constants
   from (i) the number of phases, (ii) the gap between $b_1=8192$ and $n$ (the
   paper's tightest rate uses $b_1=n$), (iii) the $F_0$ approximation, and
   (iv) the factor between the empirical gradient and the clipped/noisy version
   used internally. A large $C$ **quantifies** the looseness of the published
   constants for this Criteo instantiation; it does **not** falsify Theorem 4.2.

2. **$b_1 \ll n$.** With `b1=8192` against $n\approx 8\times10^5$, the anchor
   batch is a small fraction of the data — the bound's optimal-$b_1$ assumption
   does not hold.

3. **$q$ is not derived from the proof.** We use the default `q=30` rather than
   the phase length the analysis would prescribe.

4. **$F_0$ is approximate.** We report both $F(w_0;S)$ (upper bound, $F^*=0$)
   and $F(w_0;S)-0.4$ (approx. Criteo min BCE). The true suboptimality lies
   between.

5. **Empirical risk only.** This is Theorem 4.2; population-risk convergence
   (Theorem 4.3) is out of scope.

6. **$T$ is fixed, not swept.** The theorem's rate is derived after optimizing
   $T$; with fixed batch sizes the default `compute_noise_scales` produces noise
   scales constant in $T$, so sweeping $T$ would not trace the theorem's
   $T$-dependence. See the issue's *Out of Scope* notes.

The headline takeaways are therefore the **shape** (log-log slope vs. the
$-2/3 \to -1$ prediction) and the **fitted $C$ with its uncertainty**, read
alongside these caveats — not a verdict on the theorem's correctness.
""")

# ---------------------------------------------------------------------------
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "theorem42_stationarity_rates.ipynb"
out.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {out}  ({len(cells)} cells)")
