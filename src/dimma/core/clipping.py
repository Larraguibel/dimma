"""Per-sample gradient clipping and masking primitives.

These functions assume a per-sample pytree: every leaf has shape
``(B, *param_shape)`` where ``B`` indexes samples. They are the building
blocks for the per-sample-clipping pattern required by per-example DP
mechanisms (DP-SGD, Private SpiderBoost, etc.).

The ``+ 1e-12`` stability term in ``per_sample_clip`` is a standard
distortion accepted in DP practice; it slightly biases the clip but
prevents division-by-zero on rare all-zero gradients. This behavior is
inherited verbatim from the source implementation.

Extracted from `private_spider_boost_criteo/src/private_spiderboost.py`
without modification.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def per_sample_norms(per_sample_pytree) -> jax.Array:
    """Per-sample ``l_2`` norms across all leaves and non-batch dims.

    Parameters
    ----------
    per_sample_pytree : pytree of jax.Array
        Each leaf has shape ``(B, *param_shape)`` — the leading axis
        indexes samples.

    Returns
    -------
    norms : jax.Array, shape (B,)
        ``sqrt(sum_leaves sum_non_batch_dims (leaf ** 2))``.
    """
    leaves = jax.tree_util.tree_leaves(per_sample_pytree)
    sq = [jnp.sum(leaf.reshape(leaf.shape[0], -1) ** 2, axis=1) for leaf in leaves]
    return jnp.sqrt(sum(sq))


# Accepts a traced jax.Array — used by the variation-step kernel
def per_sample_clip(per_sample_pytree, clip_norm: float | jax.Array):
    """Clip every per-sample pytree to a global ``l_2`` norm of ``clip_norm``.

    Parameters
    ----------
    per_sample_pytree : pytree of jax.Array
        Each leaf has shape ``(B, *param_shape)``.
    clip_norm : float
        Maximum permitted global norm per sample.

    Returns
    -------
    clipped : pytree of jax.Array
        Same structure as the input. Each per-sample slice has been
        rescaled so its global ``l_2`` norm is at most ``clip_norm``.
    """
    norms = per_sample_norms(per_sample_pytree)
    factor = jnp.minimum(1.0, clip_norm / (norms + 1e-12))  # (B,)

    def _scale(leaf):
        shape = (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * factor.reshape(shape)

    return jax.tree.map(_scale, per_sample_pytree)


def per_sample_apply_mask(per_sample_pytree, mask: jax.Array):
    """Multiply each per-sample slice by ``mask[i]`` (used for Poisson masks).

    Parameters
    ----------
    per_sample_pytree : pytree of jax.Array
        Each leaf has shape ``(B, *param_shape)``.
    mask : jax.Array, shape (B,)
        Float mask, typically in {0.0, 1.0}.

    Returns
    -------
    masked : pytree of jax.Array
        Each per-sample slice has been multiplied by the mask.
    """

    def _apply(leaf):
        shape = (leaf.shape[0],) + (1,) * (leaf.ndim - 1)
        return leaf * mask.reshape(shape)

    return jax.tree.map(_apply, per_sample_pytree)
