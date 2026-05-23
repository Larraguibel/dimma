import jax
import jax.numpy as jnp
import pytest

from dimma.core import add_pytree_gaussian_noise


def test_std_zero_is_identity():
    t = {"a": jnp.arange(10.0), "b": jnp.ones((3, 4))}
    key = jax.random.PRNGKey(7)
    out = add_pytree_gaussian_noise(t, key, 0.0)
    for o, orig in zip(jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(t)):
        assert jnp.array_equal(o, orig)


def test_determinism_same_key():
    t = {"a": jnp.zeros(100), "b": jnp.zeros((5, 5))}
    key = jax.random.PRNGKey(42)
    o1 = add_pytree_gaussian_noise(t, key, 1.0)
    o2 = add_pytree_gaussian_noise(t, key, 1.0)
    for a, b in zip(jax.tree_util.tree_leaves(o1), jax.tree_util.tree_leaves(o2)):
        assert jnp.array_equal(a, b)


def test_independent_keys_difference_std():
    t = {"a": jnp.zeros((100, 100))}
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    o1 = add_pytree_gaussian_noise(t, k1, 1.0)
    o2 = add_pytree_gaussian_noise(t, k2, 1.0)
    diff = o1["a"] - o2["a"]
    s = float(jnp.std(diff))
    assert abs(s - jnp.sqrt(2.0)) / jnp.sqrt(2.0) < 0.05


def test_empirical_std_calibration():
    t = {"a": jnp.zeros((100000,))}
    key = jax.random.PRNGKey(123)
    out = add_pytree_gaussian_noise(t, key, 2.5)
    s = float(jnp.std(out["a"] - t["a"]))
    assert abs(s - 2.5) / 2.5 < 0.01
