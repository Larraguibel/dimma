"""Gaussian noise injection over JAX pytrees.

``add_pytree_gaussian_noise`` adds independent Gaussian noise of a given
standard deviation to every leaf of a pytree. The caller is responsible
for calibrating ``std`` to the mechanism's sensitivity. This module
makes no DP claims on its own — the privacy cost of a mechanism depends
on what was clipped before the noise was added, not on this function.

Extracted from `private_spider_boost_criteo/src/private_spiderboost.py`
without modification.
"""

from __future__ import annotations

import jax


def add_pytree_gaussian_noise(pytree, key: jax.Array, std: float | jax.Array):
    """Add iid ``N(0, std^2)`` noise of matching shape to every leaf.

    Parameters
    ----------
    pytree : pytree of jax.Array
        Reference pytree (the noise has the same shapes as its leaves).
    key : jax.Array
        PRNG key.
    std : float or jax.Array (scalar)
        Noise standard deviation.

    Returns
    -------
    noisy : pytree of jax.Array
        ``pytree + Gaussian noise``.
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = [
        leaf + std * jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        for leaf, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)
