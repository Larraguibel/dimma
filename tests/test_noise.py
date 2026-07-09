import jax
import jax.numpy as jnp
import pytest

from dimma.core import add_pytree_gaussian_noise, add_pytree_laplace_noise


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


# ---------------------------------------------------------------------------
# Laplace noise
# ---------------------------------------------------------------------------
def test_laplace_scale_zero_is_identity():
    t = {"a": jnp.arange(10.0), "b": jnp.ones((3, 4))}
    key = jax.random.PRNGKey(7)
    out = add_pytree_laplace_noise(t, key, 0.0)
    for o, orig in zip(jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(t)):
        assert jnp.array_equal(o, orig)


def test_laplace_determinism_same_key():
    t = {"a": jnp.zeros(100), "b": jnp.zeros((5, 5))}
    key = jax.random.PRNGKey(42)
    o1 = add_pytree_laplace_noise(t, key, 1.0)
    o2 = add_pytree_laplace_noise(t, key, 1.0)
    for a, b in zip(jax.tree_util.tree_leaves(o1), jax.tree_util.tree_leaves(o2)):
        assert jnp.array_equal(a, b)


def test_laplace_shape_dtype_preserved():
    t = {"a": jnp.zeros((4, 3), dtype=jnp.float32), "b": jnp.zeros((7,), dtype=jnp.float32)}
    key = jax.random.PRNGKey(0)
    out = add_pytree_laplace_noise(t, key, 1.5)
    for o, orig in zip(jax.tree_util.tree_leaves(out), jax.tree_util.tree_leaves(t)):
        assert o.shape == orig.shape
        assert o.dtype == orig.dtype


def test_laplace_empirical_variance():
    # Var(Lap(0, b)) = 2 b^2.  Loose tolerance for finite-sample noise.
    scale = 2.5
    t = {"a": jnp.zeros((200000,))}
    key = jax.random.PRNGKey(123)
    out = add_pytree_laplace_noise(t, key, scale)
    v = float(jnp.var(out["a"] - t["a"]))
    expected = 2.0 * scale ** 2
    assert abs(v - expected) / expected < 0.05


def test_laplace_heavier_tail_than_gaussian():
    # Excess kurtosis of a Laplace is 3 (kurtosis 6) vs 0 (kurtosis 3) for a
    # Gaussian. Standardize each sample and compare the fourth moment.
    n = 300000
    t = {"a": jnp.zeros((n,))}
    key = jax.random.PRNGKey(2024)

    lap = add_pytree_laplace_noise(t, key, 1.0)["a"]
    gau = add_pytree_gaussian_noise(t, key, 1.0)["a"]

    def kurtosis(z):
        z = z - jnp.mean(z)
        m2 = jnp.mean(z ** 2)
        m4 = jnp.mean(z ** 4)
        return float(m4 / m2 ** 2)

    k_lap = kurtosis(lap)
    k_gau = kurtosis(gau)
    # Laplace tail is heavier: its kurtosis clearly exceeds the Gaussian's.
    assert k_lap > k_gau
    assert k_lap > 4.5  # near the theoretical 6, loose lower bound
    assert abs(k_gau - 3.0) < 0.3
