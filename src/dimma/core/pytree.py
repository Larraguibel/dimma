"""Pure pytree utilities used across DP optimization primitives.

All functions in this module are dataset- and algorithm-agnostic. They
operate on JAX pytrees and have no DP-specific semantics on their own.

Extracted from `private_spider_boost_criteo/src/private_spiderboost.py`
without modification.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pytree_global_norm(pytree) -> jax.Array:
    """Global ``l_2`` norm of a pytree (across all leaves and dimensions).

    Parameters
    ----------
    pytree : pytree of jax.Array
        Any nested structure of arrays.

    Returns
    -------
    norm : jax.Array, shape ()
        ``sqrt(sum(leaf**2 for leaf in leaves))``.
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in leaves))


def pytree_sum_over_batch(per_sample_pytree):
    """Sum a per-sample pytree along the leading (batch) axis."""
    return jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0), per_sample_pytree)


def pytree_scale(pytree, scale: float | jax.Array):
    """Multiply every leaf by ``scale``."""
    return jax.tree.map(lambda leaf: leaf * scale, pytree)


def pytree_add(a, b):
    """Element-wise sum of two pytrees with identical structure."""
    return jax.tree.map(lambda x, y: x + y, a, b)


def pytree_sub(a, b):
    """Element-wise difference of two pytrees with identical structure."""
    return jax.tree.map(lambda x, y: x - y, a, b)


def pytree_zeros_like(pytree):
    """Pytree of zeros with the same structure and shapes as ``pytree``."""
    return jax.tree.map(jnp.zeros_like, pytree)
