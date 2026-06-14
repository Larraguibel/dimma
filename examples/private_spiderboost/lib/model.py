"""Reference model + non-private SPIDER baseline for the Criteo experiments.

The reference MLP and the BCE losses now live in the library
(``dimma.models``); this module re-imports them so existing notebooks and
scripts that ``import model`` keep working unchanged. Only the example-specific
pieces remain local here: the non-private ``train_spider`` baseline (governed by
ADR-0001), the ``expected_grad_evals`` budget utility, and ``visualization.py``
(its own module).
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Reference model + losses, migrated to the library (issue #8 / ADR-0002).
# Re-exported here so ``import model`` continues to expose them.
from dimma.models import (
    MLP,
    init_params,
    forward,
    per_sample_bce_loss,
    batch_bce_loss,
)
from dimma.models.mlp import _mlp_from_params

__all__ = [
    "MLP",
    "init_params",
    "forward",
    "per_sample_bce_loss",
    "batch_bce_loss",
    "_mlp_from_params",
    "expected_grad_evals",
    "SpiderTrainHistory",
    "SpiderTrainResult",
    "train_spider",
]


# ---------------------------------------------------------------------------
# Budget utilities
# ---------------------------------------------------------------------------

def expected_grad_evals(T: int, q: int, b1: int, b2: int) -> int:
    """Expected total gradient evaluations over a Private SpiderBoost run.

    The loop runs t = 0, ..., T (T+1 steps).  Anchor steps fire when
    ``t % q == 0``, costing ``b1`` gradient evaluations each (one full-batch
    gradient).  Variation steps fire otherwise, costing ``2 * b2`` each —
    one evaluation at ``params_t`` and one at ``params_prev`` — to form the
    variance-reduced SPIDER difference.  Batch sizes are the *expected*
    (nominal) sizes from the paper; no assertion is made about realised
    Poisson draws.

    Parameters
    ----------
    T : int
        Number of iterations (loop runs t = 0..T).
    q : int
        Phase length; anchor steps fire when ``t % q == 0``.
    b1 : int
        Anchor (full-gradient) batch size.
    b2 : int
        Variation batch size.

    Returns
    -------
    int
        Expected number of per-sample gradient evaluations.
    """
    n_anchors   = T // q + 1
    n_variation = T - T // q
    return n_anchors * b1 + n_variation * 2 * b2


# ---------------------------------------------------------------------------
# Non-private SPIDER training loop
# ---------------------------------------------------------------------------

class SpiderTrainHistory(NamedTuple):
    """Per-step record from train_spider.

    Attributes
    ----------
    grad_norm : list[float], length T+1
        Global L2 norm of the gradient estimate at each step.
    wall_time_s : list[float], length T+1
        Wall-clock seconds per step.
    output_step : int
        Step t* uniformly drawn from {1, ..., T} at which params_random
        was snapshotted (mirrors the DP random-output rule).
    """
    grad_norm: list
    wall_time_s: list
    output_step: int


class SpiderTrainResult(NamedTuple):
    """Output of train_spider.

    Attributes
    ----------
    params_final : pytree
        Parameters after step T.
    params_random : pytree
        Parameters at the uniformly random output step t* ∈ {1, ..., T}.
        Mirrors the output rule of Private SpiderBoost for fair comparison.
    history : SpiderTrainHistory
    """
    params_final: Any
    params_random: Any
    history: SpiderTrainHistory


def train_spider(
    x_train: jax.Array,
    y_train: jax.Array,
    init_params: Any,
    T: int,
    q: int,
    b1: int,
    b2: int,
    eta: float,
    seed: int,
) -> SpiderTrainResult:
    """Non-private SPIDER training loop with the same anchor/variation step
    structure as Private SpiderBoost, but without gradient clipping or noise.

    Anchor steps (t mod q == 0) compute a fresh mean gradient over a batch
    of size b1. Variation steps (t mod q != 0) accumulate the mean gradient
    difference over a batch of size b2, giving a variance-reduced estimate.
    Parameters are updated via SGD: w_{t+1} = w_t - eta * grad_estimate_t.

    The random output step t* is drawn from {1, ..., T} with the same RNG
    convention as dimma.train so that params_random is directly comparable
    to the DP run's params_random.

    Parameters
    ----------
    x_train, y_train : jax.Array
        Full training set.
    init_params : pytree
        Initial parameters (e.g. from model.init_params).
    T : int
        Number of iterations. The loop runs t = 0, ..., T.
    q : int
        Phase length. Anchor steps fire when t mod q == 0.
    b1, b2 : int
        Anchor and variation batch sizes (sampled without replacement).
    eta : float
        Learning rate.
    seed : int
        Master seed. Sampling uses JAX PRNG; output-step draw uses
        numpy.random.default_rng(seed + 7919) to match dimma.train.

    Returns
    -------
    SpiderTrainResult
    """
    sampling_key = jax.random.PRNGKey(seed)
    control_rng = np.random.default_rng(seed + 7919)
    n = int(x_train.shape[0])

    @jax.jit
    def anchor_grad(params, x_batch, y_batch):
        return jax.grad(batch_bce_loss)(params, x_batch, y_batch)

    @jax.jit
    def variation_delta(params, params_prev, x_batch, y_batch):
        g_t = jax.grad(batch_bce_loss)(params, x_batch, y_batch)
        g_prev = jax.grad(batch_bce_loss)(params_prev, x_batch, y_batch)
        return jax.tree.map(lambda a, b: a - b, g_t, g_prev)

    def _norm(tree):
        leaves = jax.tree.leaves(tree)
        return float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves)))

    output_step = int(control_rng.integers(low=1, high=T + 1))
    params = init_params
    params_prev = init_params
    grad_est: Any = None
    params_random = None

    grad_norm_hist: list = []
    wall_time_hist: list = []

    for t in range(T + 1):
        t_start = time.perf_counter()
        sampling_key, subkey = jax.random.split(sampling_key)
        is_anchor = (t % q == 0)

        if is_anchor:
            idx = np.asarray(
                jax.random.choice(subkey, n, shape=(min(b1, n),), replace=False)
            )
            x_b, y_b = x_train[idx], y_train[idx]
            grad_est = anchor_grad(params, x_b, y_b)
        else:
            idx = np.asarray(
                jax.random.choice(subkey, n, shape=(min(b2, n),), replace=False)
            )
            x_b, y_b = x_train[idx], y_train[idx]
            delta = variation_delta(params, params_prev, x_b, y_b)
            grad_est = jax.tree.map(lambda a, b: a + b, grad_est, delta)

        params_prev = params
        params = jax.tree.map(lambda p, g: p - eta * g, params, grad_est)

        if t == output_step:
            params_random = params

        grad_norm_hist.append(_norm(grad_est))
        wall_time_hist.append(time.perf_counter() - t_start)

    assert params_random is not None, (
        "Internal error: output_step was never reached. "
        "Check loop range and output_step draw."
    )

    return SpiderTrainResult(
        params_final=params,
        params_random=params_random,
        history=SpiderTrainHistory(
            grad_norm=grad_norm_hist,
            wall_time_s=wall_time_hist,
            output_step=output_step,
        ),
    )
