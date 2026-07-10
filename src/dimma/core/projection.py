"""Euclidean projection onto the ``l_1`` ball over JAX arrays and pytrees.

``project_l1_ball`` computes the Euclidean projection of a vector onto the
``l_1`` ball ``{ x : ‖x‖_1 <= radius }`` using the sort-based algorithm of
Duchi et al. (ICML 2008), an ``O(d log d)`` operation. ``project_l1_ball_pytree``
applies the same projection *globally* across every leaf of a pytree by
flattening into a single vector, projecting, and unflattening.

These functions are purely geometric. This module makes no DP claims on its
own — the ``l_1``-ball projection is a deterministic post-processing step; its
role in a differentially private mechanism (calibrating ``radius`` to a
sparsity bound, applying it after calibrated noise) lives in higher layers.

The implementation is fully branchless — no data-dependent shapes — so it is
``jit``-safe even when ``radius`` is a *traced* value computed at runtime.
Reference: J. Duchi, S. Shalev-Shwartz, Y. Singer, T. Chandra, "Efficient
Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def project_l1_ball(x: jax.Array, radius: float | jax.Array) -> jax.Array:
    """Euclidean projection of ``x`` onto the ``l_1`` ball of a given radius.

    Solves ``argmin_z ‖z − x‖_2`` subject to ``‖z‖_1 <= radius`` via the
    sort-based Duchi et al. (2008) algorithm: sort ``|x|`` descending, form
    the cumulative sum, locate the soft-threshold ``theta``, then apply
    ``sign(x) * max(|x| − theta, 0)``.

    The computation uses only fixed-shape operations, so it is ``jit``-safe
    with a traced ``radius``. Vectors already inside the ball are returned
    **bit-exactly unchanged**.

    Parameters
    ----------
    x : jax.Array
        Input vector (1-D). Its ``l_1`` geometry is projected onto the ball.
    radius : float or jax.Array (scalar)
        Non-negative ball radius. May be a traced value.

    Returns
    -------
    projected : jax.Array
        Same shape and dtype as ``x``; satisfies ``‖projected‖_1 <= radius``.
    """
    # A negative radius describes an empty l_1 ball and would silently project
    # to the origin. Guard it eagerly, but ONLY when radius is concrete: this
    # function is jit-traced with a *traced* radius in the SpiderBoost kernels,
    # where a Python comparison would break tracing. Skipping the check under a
    # tracer adds no ops to the jaxpr, so the traced program is unchanged.
    if not isinstance(radius, jax.core.Tracer):
        assert radius >= 0, f"radius must be non-negative, got {radius}."

    abs_x = jnp.abs(x)
    d = x.shape[0]

    # Sort |x| in descending order and form the running cumulative sum.
    u = jnp.sort(abs_x)[::-1]
    cssv = jnp.cumsum(u)

    # rho = number of coordinates k (1-indexed) with u_k * k > cssv_k − radius.
    k = jnp.arange(1, d + 1, dtype=cssv.dtype)
    rho = jnp.sum(u * k > (cssv - radius))
    rho = jnp.maximum(rho, 1)  # guard the all-zero case (avoids rho = 0)

    theta = jnp.maximum((cssv[rho - 1] - radius) / rho, 0.0)
    projected = jnp.sign(x) * jnp.maximum(abs_x - theta, 0.0)

    # Inputs already inside the ball are returned bit-exactly unchanged.
    l1norm = jnp.sum(abs_x)
    return jnp.where(l1norm <= radius, x, projected)


def project_l1_ball_pytree(pytree: Any, radius: float | jax.Array) -> Any:
    """Project a whole pytree onto a single global ``l_1`` ball.

    All leaves are flattened into one vector (via ``ravel_pytree``), projected
    together onto the ``l_1`` ball of the given ``radius``, then unflattened
    back into the original structure. The constraint ``‖·‖_1 <= radius`` is
    therefore enforced across the *concatenation* of every leaf, not per-leaf.

    Parameters
    ----------
    pytree : pytree of jax.Array
        Any nested structure of arrays.
    radius : float or jax.Array (scalar)
        Non-negative ball radius. May be a traced value.

    Returns
    -------
    projected : pytree of jax.Array
        Same structure and leaf shapes as ``pytree``; the flattened vector
        satisfies ``‖·‖_1 <= radius``.
    """
    flat, unravel = ravel_pytree(pytree)
    return unravel(project_l1_ball(flat, radius))
