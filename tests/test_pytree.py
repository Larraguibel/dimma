import jax
import jax.numpy as jnp
import pytest

from dimma.core import (
    pytree_global_norm,
    pytree_sub,
    pytree_zeros_like,
    pytree_scale,
    pytree_sum_over_batch,
)


def test_pytree_global_norm_3_4_5():
    tree = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([[0.0, 0.0]])}
    assert float(pytree_global_norm(tree)) == pytest.approx(5.0, abs=1e-6)


def test_pytree_sub_self_is_zero():
    t = {"a": jnp.array([1.0, 2.0, 3.0]), "b": jnp.ones((2, 3))}
    d = pytree_sub(t, t)
    for leaf in jax.tree_util.tree_leaves(d):
        assert jnp.all(leaf == 0.0)


def test_pytree_zeros_like_shape():
    t = {"a": jnp.ones((2, 3)), "b": jnp.arange(5.0)}
    z = pytree_zeros_like(t)
    leaves_t = jax.tree_util.tree_leaves(t)
    leaves_z = jax.tree_util.tree_leaves(z)
    assert len(leaves_t) == len(leaves_z)
    for lt, lz in zip(leaves_t, leaves_z):
        assert lt.shape == lz.shape
        assert jnp.all(lz == 0.0)


def test_pytree_scale_composition():
    t = {"a": jnp.array([1.0, 2.0]), "b": jnp.ones((2, 2))}
    scaled = pytree_scale(pytree_scale(t, 2.0), 3.0)
    for orig, s in zip(jax.tree_util.tree_leaves(t), jax.tree_util.tree_leaves(scaled)):
        assert jnp.allclose(s, 6.0 * orig, atol=1e-6)


def test_pytree_sum_over_batch_shape():
    t = {"a": jnp.ones((4, 3)), "b": jnp.ones((4, 2, 5))}
    s = pytree_sum_over_batch(t)
    assert s["a"].shape == (3,)
    assert s["b"].shape == (2, 5)
