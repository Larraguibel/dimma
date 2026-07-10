"""Private SpiderBoost training loop (Algorithm 2, Arora et al. 2023).

Generalized over loss, model, and data. The library does no model
construction, no metric computation, no I/O — those concerns are
delegated to the caller via ``init_params`` and ``step_callback``.

Loop indexing follows the paper literally:
- iterations t = 0, ..., T (i.e. T+1 calls to the step kernels),
- anchor branch when t mod q == 0 (so anchors at 0, q, 2q, ...),
- random output rule uniform over {w_1, ..., w_T}.

The privacy budget calibrated by ``compute_noise_scales`` matches this
loop length and indexing convention exactly. See Notion for the
discussion of the T-vs-T+1 question relative to Theorem B.2 of the
paper.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Literal, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from dimma.accounting.spiderboost import NoiseScales
from dimma.algorithms.spiderboost.kernels import (
    StepOutput,
    make_anchor_step,
    make_variation_step,
    sgd_update,
)
from dimma.core.pytree import pytree_global_norm, pytree_sub
from dimma.core.sampling.poisson import (
    poisson_padded_batch_size,
    poisson_subsample,
    poisson_subsample_truncated,
)


class TrainConfig(NamedTuple):
    """Configuration for ``train``.

    Privacy and algorithm parameters only. No model-specific or
    evaluation-specific fields — those are the caller's responsibility.

    Attributes
    ----------
    epsilon, delta : float
        Target privacy budget. Used to compute ``NoiseScales`` via
        ``compute_noise_scales``; the caller must do this and pass
        the result into ``train``.
    L0 : float
        Per-sample gradient clipping bound (anchor step sensitivity).
    L1 : float
        Lipschitz constant of the gradient (variation step sensitivity
        bound; ``clip_c = L1 * ||w_t - w_{t-1}||``).
    T : int
        Number of training iterations. The loop runs t = 0, ..., T,
        which is T+1 step-kernel invocations. See the module docstring
        for the indexing convention.
    q : int
        Phase length. Anchor steps fire when ``t mod q == 0``.
    b1, b2 : int
        Expected anchor and variation batch sizes (Poisson). The
        averaging in the kernels divides by these expected sizes, not
        by realized batch sizes — this is the standard DP convention.
    eta : float
        Learning rate.
    seed : int
        Master seed. Two NumPy RNGs and one JAX PRNGKey are derived from
        this. See ``train``'s docstring for the RNG split.
    margin_sigmas : float
        Safety margin for ``poisson_padded_batch_size``. Defaults to
        the value used in the source implementation (6.0).
    s : int or None
        Gradient sparsity for optional ``l_1``-ball projection
        post-processing in the step kernels. ``None`` (default) disables
        projection — the kernels run the unmodified SpiderBoost step. When
        set, the anchor estimate is projected onto ``B_1(0, L0*sqrt(s))``
        and the variation increment onto ``B_1(0, L1*delta_w*sqrt(2*s))``
        (Ghazi et al. 2024, Algorithm 1). Projection is post-processing;
        the privacy accounting (``NoiseScales``) is UNCHANGED.
    """
    epsilon: float
    delta: float
    L0: float
    L1: float
    T: int
    q: int
    b1: int
    b2: int
    eta: float
    seed: int
    margin_sigmas: float = 6.0
    s: int | None = None


class StepInfo(NamedTuple):
    """Per-step information passed to ``step_callback``.

    Attributes
    ----------
    step : int
        The current iteration index t, in 0, ..., T.
    is_anchor : bool
        True if this step was an anchor step (t mod q == 0).
    params : Any
        Model parameters AFTER this step's update.
    grad_estimate : Any
        Algorithm 2's running gradient estimate ∇_t.
    grad_norm : float
        Global L2 norm of ``grad_estimate``.
    delta_w : float
        ||w_t - w_{t-1}|| for variation steps; ``float('nan')`` for
        anchor steps (delta_w is not defined for anchors).
    realized_noise_std : float
        sigma1 for anchor steps; ``min(sigma2 * delta_w, sigma2_hat)``
        for variation steps.

    Privacy warning
    ---------------
    ``params`` and ``grad_estimate`` are post-mechanism outputs of a
    DP-trained model; they are safe to inspect under the calibrated
    (epsilon, delta) bound. They are NOT safe to combine with raw
    training data, to log per-sample, or to derive secondary statistics
    that re-introduce dependence on individual examples. The library
    cannot enforce this — the callback writer is responsible.
    """
    step: int
    is_anchor: bool
    params: Any
    grad_estimate: Any
    grad_norm: float
    delta_w: float
    realized_noise_std: float


class TrainHistory(NamedTuple):
    """Per-step record of training. No model-specific metrics.

    Attributes
    ----------
    grad_norm : list[float], length T+1
        ``||grad_estimate||_2`` at each step.
    wall_time_s : list[float], length T+1
        Wall-clock seconds per step.
    noise_scales : NoiseScales
        The ``NoiseScales`` used to drive this run (echoed for
        reproducibility).
    output_step : int
        The step index t* uniformly sampled from {1, ..., T} at which
        ``params_random`` was snapshotted.
    """
    grad_norm: list
    wall_time_s: list
    noise_scales: NoiseScales
    output_step: int


class TrainResult(NamedTuple):
    """The output of ``train``.

    Attributes
    ----------
    params_final : Any
        The parameters at the end of the loop (after step T).
    params_random : Any
        The parameters at the random output step t* ∈ {1, ..., T}.
        This is the iterate returned by Algorithm 2's output rule.
    history : TrainHistory
    """
    params_final: Any
    params_random: Any
    history: TrainHistory


def train(
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    per_sample_loss_fn: Callable,
    init_params: Any,
    config: TrainConfig,
    noise_scales: NoiseScales,
    sampler: Literal["poisson", "poisson_truncated"] = "poisson",
    step_callback: Optional[Callable[[StepInfo], None]] = None,
) -> TrainResult:
    """Run Private SpiderBoost (Algorithm 2 of Arora et al. 2023).

    Parameters
    ----------
    x_train, y_train : jnp.ndarray
        Training data. The arrays are indexed by the Poisson sampler's
        ``indices`` array each step; both must support fancy indexing
        on the leading axis. ``y_train`` is passed through verbatim to
        ``per_sample_loss_fn``.
    per_sample_loss_fn : callable
        ``per_sample_loss_fn(params, x_single, y_single) -> scalar``.
        Required, no default. The library makes no assumption about
        the loss; it is what gets privatized. The caller is responsible
        for ensuring this function's per-sample gradient is bounded by
        ``config.L0`` after clipping (which the kernel enforces).
    init_params : pytree
        Initial parameters. Any JAX-compatible pytree.
    config : TrainConfig
    noise_scales : NoiseScales
        Computed by ``dimma.accounting.spiderboost.compute_noise_scales``.
        The caller passes this in explicitly so that the privacy
        accounting decision is visible at the call site.
    sampler : {"poisson", "poisson_truncated"}, default "poisson"
        Subsampling strategy. ``"poisson"`` raises on oversize batches;
        ``"poisson_truncated"`` deterministically truncates and uses
        the heuristic accountant. See ``dimma.core.sampling.poisson``
        for details.
    step_callback : callable or None
        Called once per step with a ``StepInfo``. The library does no
        printing, evaluation, or metric computation. Use the callback
        for whatever per-step bookkeeping you need.

        Privacy: read the ``StepInfo`` privacy warning in the
        ``StepInfo`` docstring. The library cannot enforce safe
        callback behavior.

    Returns
    -------
    TrainResult

    RNG
    ---
    Three independent RNGs are derived from ``config.seed``:

    - ``sampling_rng`` (numpy.random.Generator, seeded with
      ``config.seed``): drives Poisson mask sampling. Privacy-relevant.
    - ``noise_key`` (jax.random.PRNGKey, seeded with
      ``config.seed + 1``): drives Gaussian noise injection.
      Privacy-relevant.
    - ``control_rng`` (numpy.random.Generator, seeded with
      ``config.seed + 7919``): drives the random-output-step draw.
      NOT privacy-relevant.

    Notes
    -----
    Per-sample gradients are computed via ``jax.vmap(jax.grad(
    per_sample_loss_fn), in_axes=(None, 0, 0))``. The kernel JIT-compiles
    this function once per (anchor/variation) branch, with the
    per-sample gradient function as a static argument so the
    compilation key is stable across the loop.
    """
    sampling_rng = np.random.default_rng(config.seed)
    control_rng = np.random.default_rng(config.seed + 7919)
    noise_key = jax.random.PRNGKey(config.seed + 1)

    n = int(x_train.shape[0])
    p1 = config.b1 / n
    p2 = config.b2 / n
    b1_max = poisson_padded_batch_size(config.b1, n, config.margin_sigmas)
    b2_max = poisson_padded_batch_size(config.b2, n, config.margin_sigmas)

    if sampler == "poisson":
        sample_fn = poisson_subsample
    elif sampler == "poisson_truncated":
        sample_fn = poisson_subsample_truncated
    else:
        raise ValueError(
            f"Unknown sampler: {sampler!r}. "
            "Expected 'poisson' or 'poisson_truncated'."
        )

    per_sample_grad_fn = jax.vmap(
        jax.grad(per_sample_loss_fn), in_axes=(None, 0, 0)
    )

    anchor_step_jit = jax.jit(make_anchor_step(per_sample_grad_fn, s=config.s))
    variation_step_jit = jax.jit(make_variation_step(per_sample_grad_fn, s=config.s))

    params = init_params
    params_prev = init_params
    grad_estimate = None

    output_step = int(control_rng.integers(low=1, high=config.T + 1))
    params_random = None

    grad_norm_hist: list = []
    wall_time_hist: list = []

    for t in range(config.T + 1):
        t_start = time.perf_counter()
        is_anchor = (t % config.q == 0)

        if is_anchor:
            indices, mask_np = sample_fn(sampling_rng, n, p1, b1_max)
            x_batch = jnp.asarray(x_train[indices])
            y_batch = jnp.asarray(y_train[indices])
            mask = jnp.asarray(mask_np)
            noise_key, sub = jax.random.split(noise_key)
            out: StepOutput = anchor_step_jit(
                params, x_batch, y_batch, mask,
                config.b1, config.L0, noise_scales.sigma1, sub,
            )
            delta_w_val = float("nan")
            realized_std = float(noise_scales.sigma1)
        else:
            indices, mask_np = sample_fn(sampling_rng, n, p2, b2_max)
            x_batch = jnp.asarray(x_train[indices])
            y_batch = jnp.asarray(y_train[indices])
            mask = jnp.asarray(mask_np)
            noise_key, sub = jax.random.split(noise_key)
            out = variation_step_jit(
                params, params_prev, grad_estimate, x_batch, y_batch, mask,
                config.b2, config.L1, noise_scales.sigma2,
                noise_scales.sigma2_hat, sub,
            )
            delta_w_val = float(pytree_global_norm(pytree_sub(params, params_prev)))
            realized_std = float(jnp.minimum(
                noise_scales.sigma2 * delta_w_val, noise_scales.sigma2_hat
            ))

        grad_estimate = out.grad_estimate
        params_prev = params
        params = sgd_update(params, grad_estimate, config.eta)

        if t == output_step:
            params_random = params

        g_norm = float(out.grad_norm)
        grad_norm_hist.append(g_norm)
        wall_time_hist.append(time.perf_counter() - t_start)

        if step_callback is not None:
            step_callback(StepInfo(
                step=t,
                is_anchor=is_anchor,
                params=params,
                grad_estimate=grad_estimate,
                grad_norm=g_norm,
                delta_w=delta_w_val,
                realized_noise_std=realized_std,
            ))

    assert params_random is not None, (
        "Internal error: output_step was not assigned. "
        "Check the output_step draw and loop range."
    )

    return TrainResult(
        params_final=params,
        params_random=params_random,
        history=TrainHistory(
            grad_norm=grad_norm_hist,
            wall_time_s=wall_time_hist,
            noise_scales=noise_scales,
            output_step=output_step,
        ),
    )
