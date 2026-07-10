# FROZEN SNAPSHOT — do not edit.
# Verbatim copy of src/dimma/algorithms/spiderboost/kernels.py as of the commit
# immediately BEFORE the issue #24 deduplication refactor (HEAD 90a83bf,
# "Eagerly validate the sparsity hyperparameter s"). It is the ground-truth
# reference used by tests/test_spiderboost_jaxpr.py: the refactored s=None
# kernels must trace to a jaxpr byte-identical to the ones these produce.
# Because both are traced under the same JAX version at test time, the
# comparison is immune to jaxpr-text drift across JAX releases.
"""Private SpiderBoost step kernels (Algorithm 2 of Arora et al., ICML 2023).

This module contains only the mathematical kernels for one anchor step
and one variation step. Subsampling, clipping decisions, and the outer
loop live elsewhere. The factory pattern (``make_anchor_step``,
``make_variation_step``) takes a per-sample gradient function and returns
a step kernel closed over it; the resulting kernel is suitable for
``jax.jit`` at the call site.

Reference
---------
R. Arora, R. Bassily, T. González, C. Guzmán, M. Menart, E. Ullah,
"Faster Rates of Convergence to Stationary Points in Differentially
Private Optimization", *Proceedings of the 40th International Conference
on Machine Learning*, 2023.

Extracted from ``private_spider_boost_criteo/src/private_spiderboost.py``
without modification, save for import paths.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from dimma.core.pytree import (
    pytree_global_norm,
    pytree_sub,
    pytree_add,
    pytree_scale,
    pytree_sum_over_batch,
)
from dimma.core.clipping import per_sample_clip, per_sample_apply_mask
from dimma.core.noise import add_pytree_gaussian_noise
from dimma.core.projection import project_l1_ball_pytree


def _validate_s(s: int | None) -> None:
    """Eagerly reject an invalid sparsity ``s`` at factory setup time.

    ``s`` is a *static* Python int used compile-time as ``sqrt(s)`` (anchor)
    and ``sqrt(2 * s)`` (variation), so the guard runs here — outside anything
    traced — to keep the ``s=None`` program byte-identical. Consistent with the
    standalone ``projection_mechanism``, which enforces ``s >= 1``. Note that
    ``isinstance(True, int)`` is ``True`` in Python, so ``bool`` is rejected
    explicitly: ``s=True`` is not a valid sparsity.
    """
    if s is None:
        return
    if isinstance(s, bool) or not isinstance(s, int) or s < 1:
        raise ValueError(f"s must be a positive integer or None, got {s!r}.")


class StepOutput(NamedTuple):
    """Return value of one SpiderBoost step.

    Attributes
    ----------
    grad_estimate : pytree
        New running gradient estimate ``∇_t``.
    grad_norm : jax.Array, shape ()
        Global ``l_2`` norm of ``grad_estimate`` (logged every step).
    """

    grad_estimate: Any
    grad_norm: jax.Array


def make_anchor_step(per_sample_grad_fn, s: int | None = None):
    """Build an anchor-step kernel for a given per-sample gradient function.

    Parameters
    ----------
    per_sample_grad_fn : callable
        Vmapped per-sample gradient: ``(params, x_batch, y_batch) -> pytree``
        with leaves of shape ``(B, *param_shape)``.
    s : int or None, optional
        Gradient sparsity for optional ``l_1``-ball projection post-processing.
        When ``None`` (default) the returned kernel is the *unmodified*
        SpiderBoost anchor step — byte-identical XLA program to the pre-Ghazi
        implementation. When an ``int``, the kernel additionally projects the
        noisy anchor estimate onto the ``l_1``-ball of radius ``L0 * sqrt(s)``
        (the Ghazi et al. 2024, Algorithm 1 relaxation of the sparse set). ``s``
        is a *static* Python int, so ``sqrt(s)`` is a compile-time constant.

    Returns
    -------
    anchor_step : callable
        ``anchor_step(params, x_batch, y_batch, mask, b1, L0, sigma1, key)
        -> StepOutput``. ``mask`` is a (B,) Poisson mask; ``b1`` is the
        *expected* batch size used for averaging.

    Notes
    -----
    Algorithm 2, anchor branch (``t mod q == 0``)::

        g_t  ~ N(0, sigma1^2 I)
        ∇_t  = (1 / b1) * sum_{x in S_t} clip(∇f(w_t; x), L0)  +  g_t
    """
    _validate_s(s)

    def anchor_step(params, x_batch, y_batch, mask, b1, L0, sigma1, key):
        per_sample = per_sample_grad_fn(params, x_batch, y_batch)
        per_sample = per_sample_clip(per_sample, L0)            # Algorithm 2, Section 4.1
        per_sample = per_sample_apply_mask(per_sample, mask)    # Poisson subsampling
        summed = pytree_sum_over_batch(per_sample)
        averaged = pytree_scale(summed, 1.0 / b1)
        noisy = add_pytree_gaussian_noise(averaged, key, sigma1)
        return StepOutput(grad_estimate=noisy, grad_norm=pytree_global_norm(noisy))

    if s is None:
        return anchor_step

    def anchor_step_projected(params, x_batch, y_batch, mask, b1, L0, sigma1, key):
        """Anchor step with ``l_1``-ball projection post-processing.

        Identical to ``anchor_step`` up to the noise injection, then projects
        the noisy anchor estimate onto ``K = B_1(0, L0 * sqrt(s))`` — the
        Ghazi et al. (2024), Algorithm 1 convex relaxation of the s-sparse set
        (a clipped s-sparse gradient has ``||.||_1 <= sqrt(s) * ||.||_2 <=
        L0 * sqrt(s)``). The projection is deterministic **post-processing** of a
        DP quantity, so the privacy accounting is UNCHANGED (Ghazi et al. 2024).
        """
        per_sample = per_sample_grad_fn(params, x_batch, y_batch)
        per_sample = per_sample_clip(per_sample, L0)            # Algorithm 2, Section 4.1
        per_sample = per_sample_apply_mask(per_sample, mask)    # Poisson subsampling
        summed = pytree_sum_over_batch(per_sample)
        averaged = pytree_scale(summed, 1.0 / b1)
        noisy = add_pytree_gaussian_noise(averaged, key, sigma1)
        radius = L0 * math.sqrt(s)  # static sparsity -> static float; L0 traced
        noisy = project_l1_ball_pytree(noisy, radius)           # post-processing (Ghazi 2024)
        return StepOutput(grad_estimate=noisy, grad_norm=pytree_global_norm(noisy))

    return anchor_step_projected


def make_variation_step(per_sample_grad_fn, s: int | None = None):
    """Build a variation-step kernel for a given per-sample gradient function.

    Parameters
    ----------
    per_sample_grad_fn : callable
        Vmapped per-sample gradient: ``(params, x_batch, y_batch) -> pytree``
        with leaves of shape ``(B, *param_shape)``.
    s : int or None, optional
        Gradient sparsity for optional ``l_1``-ball projection post-processing.
        When ``None`` (default) the returned kernel is the *unmodified*
        SpiderBoost variation step — byte-identical XLA program to the pre-Ghazi
        implementation. When an ``int``, the kernel projects the noisy SPIDER
        *increment* ``Δ_t`` (not the accumulated estimate) onto the ``l_1``-ball
        of radius ``L1 * delta_w * sqrt(2 * s)`` before accumulation: a
        difference of two s-sparse vectors is at most 2s-sparse and, once
        clipped to ``l_2 <= L1 * delta_w``, has ``l_1 <= sqrt(2s) * L1 *
        delta_w`` (Ghazi et al. 2024, Algorithm 1 relaxation). ``sqrt(2 * s)``
        uses the static int ``s``; ``delta_w`` and ``L1`` are traced runtime
        values (the projection accepts a traced radius).

    Returns
    -------
    variation_step : callable
        ``variation_step(params_t, params_prev, prev_grad_est, x_batch,
        y_batch, mask, b2, L1, sigma2, sigma2_hat, key) -> StepOutput``.

    Notes
    -----
    Algorithm 2, variation branch (``t mod q != 0``)::

        delta_w  = ||w_t - w_{t-1}||
        clip_c   = L1 * delta_w
        g_t      ~ N(0, min(sigma2 * delta_w, sigma2_hat)^2 I)
        Δ_t      = (1 / b2) * sum_{x in S_t}
                   clip(∇f(w_t; x) - ∇f(w_{t-1}; x), clip_c)  +  g_t
        ∇_t      = ∇_{t-1} + Δ_t
    """
    _validate_s(s)

    def variation_step(params_t, params_prev, prev_grad_est, x_batch, y_batch,
                       mask, b2, L1, sigma2, sigma2_hat, key):
        delta_w = pytree_global_norm(pytree_sub(params_t, params_prev))

        per_sample_t = per_sample_grad_fn(params_t, x_batch, y_batch)
        per_sample_prev = per_sample_grad_fn(params_prev, x_batch, y_batch)
        per_sample_diff = pytree_sub(per_sample_t, per_sample_prev)

        clip_c = L1 * delta_w
        per_sample_diff = per_sample_clip(per_sample_diff, clip_c)   # Algorithm 2, Section 4.1
        per_sample_diff = per_sample_apply_mask(per_sample_diff, mask)
        summed = pytree_sum_over_batch(per_sample_diff)
        averaged = pytree_scale(summed, 1.0 / b2)

        noise_std = jnp.minimum(sigma2 * delta_w, sigma2_hat)
        noisy_delta = add_pytree_gaussian_noise(averaged, key, noise_std)

        new_grad_est = pytree_add(prev_grad_est, noisy_delta)
        return StepOutput(
            grad_estimate=new_grad_est,
            grad_norm=pytree_global_norm(new_grad_est),
        )

    if s is None:
        return variation_step

    def variation_step_projected(params_t, params_prev, prev_grad_est, x_batch,
                                 y_batch, mask, b2, L1, sigma2, sigma2_hat, key):
        """Variation step with ``l_1``-ball projection of the SPIDER increment.

        Identical to ``variation_step`` up to the noisy increment ``Δ_t``, which
        is then projected onto ``K = B_1(0, L1 * delta_w * sqrt(2 * s))`` BEFORE
        the accumulation ``∇_t = ∇_{t-1} + Δ_t`` — the sparse object is the
        *increment*, not the accumulated estimate. The projection is
        deterministic **post-processing** of a DP quantity, so the privacy
        accounting is UNCHANGED (Ghazi et al. 2024).
        """
        delta_w = pytree_global_norm(pytree_sub(params_t, params_prev))

        per_sample_t = per_sample_grad_fn(params_t, x_batch, y_batch)
        per_sample_prev = per_sample_grad_fn(params_prev, x_batch, y_batch)
        per_sample_diff = pytree_sub(per_sample_t, per_sample_prev)

        clip_c = L1 * delta_w
        per_sample_diff = per_sample_clip(per_sample_diff, clip_c)   # Algorithm 2, Section 4.1
        per_sample_diff = per_sample_apply_mask(per_sample_diff, mask)
        summed = pytree_sum_over_batch(per_sample_diff)
        averaged = pytree_scale(summed, 1.0 / b2)

        noise_std = jnp.minimum(sigma2 * delta_w, sigma2_hat)
        noisy_delta = add_pytree_gaussian_noise(averaged, key, noise_std)

        radius = L1 * delta_w * math.sqrt(2 * s)  # static 2s; L1, delta_w traced
        noisy_delta = project_l1_ball_pytree(noisy_delta, radius)   # post-processing (Ghazi 2024)

        new_grad_est = pytree_add(prev_grad_est, noisy_delta)
        return StepOutput(
            grad_estimate=new_grad_est,
            grad_norm=pytree_global_norm(new_grad_est),
        )

    return variation_step_projected


def sgd_update(params, grad_estimate, lr: float):
    """Apply one ``params <- params - lr * grad`` update.

    Parameters
    ----------
    params : pytree
    grad_estimate : pytree
        Same structure as ``params``.
    lr : float
        Learning rate.

    Returns
    -------
    new_params : pytree
    """
    return jax.tree.map(lambda p, g: p - lr * g, params, grad_estimate)
