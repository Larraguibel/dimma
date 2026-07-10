"""Gaussian and Laplace noise injection over JAX pytrees.

``add_pytree_gaussian_noise`` and ``add_pytree_laplace_noise`` add
independent noise of a given scale to every leaf of a pytree. The caller
is responsible for calibrating the scale to the mechanism's sensitivity.
This module makes no DP claims on its own — the privacy cost of a
mechanism depends on what was clipped before the noise was added, not on
these functions.

The public API and traced behavior of ``add_pytree_gaussian_noise`` are
preserved exactly from
`private_spider_boost_criteo/src/private_spiderboost.py` (verified by the
regression oracle in ``tests/test_regression_against_source.py``); the
per-leaf sampling is now routed through the shared private helper
``_add_pytree_noise``.
"""

from __future__ import annotations

import jax


def _add_pytree_noise(pytree, key: jax.Array, scale: float | jax.Array, sample_fn):
    """Add iid noise of matching shape to every leaf via ``sample_fn``.

    ``sample_fn`` is a ``jax.random`` sampler ``(key, shape, dtype) -> array``
    (e.g. ``jax.random.normal`` or ``jax.random.laplace``). The traced program
    is identical to inlining the sampler: same key split, same per-leaf
    ``leaf + scale * sample_fn(k, leaf.shape, dtype=leaf.dtype)`` in the same
    order. Shared by the two public wrappers below.
    """
    # The module docstring's provenance claim was updated when this helper was
    # extracted (#25): the traced program is unchanged (same key splits, same
    # ops), enforced by tests/test_regression_against_source.py.
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    keys = jax.random.split(key, len(leaves))
    noisy_leaves = [
        leaf + scale * sample_fn(k, leaf.shape, dtype=leaf.dtype)
        for leaf, k in zip(leaves, keys)
    ]
    return jax.tree_util.tree_unflatten(treedef, noisy_leaves)


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
    return _add_pytree_noise(pytree, key, std, jax.random.normal)


def add_pytree_laplace_noise(pytree, key: jax.Array, scale: float | jax.Array):
    """Add iid ``Lap(0, scale)`` noise of matching shape to every leaf.

    ``scale`` is the Laplace ``b`` parameter (the density is proportional to
    ``exp(-|x| / b)``), **not** the standard deviation. The variance of each
    coordinate is ``2 * scale ** 2``. The caller is responsible for
    calibrating ``scale`` to the mechanism's ``l_1`` sensitivity; this
    function makes no DP claims on its own.

    Parameters
    ----------
    pytree : pytree of jax.Array
        Reference pytree (the noise has the same shapes as its leaves).
    key : jax.Array
        PRNG key.
    scale : float or jax.Array (scalar)
        Laplace ``b`` parameter (not the standard deviation).

    Returns
    -------
    noisy : pytree of jax.Array
        ``pytree + Laplace noise``.
    """
    return _add_pytree_noise(pytree, key, scale, jax.random.laplace)
