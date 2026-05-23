import jax
import jax.numpy as jnp
import pytest

from dimma.core import per_sample_norms, per_sample_clip, per_sample_apply_mask


def test_per_sample_norms_matches_manual():
    leaf = jnp.arange(12).reshape(3, 4).astype(jnp.float32)
    t = {"w": leaf}
    norms = per_sample_norms(t)
    expected = jnp.linalg.norm(leaf, axis=1)
    assert jnp.allclose(norms, expected, atol=1e-5)


def test_per_sample_clip_enforces_bound():
    key = jax.random.PRNGKey(0)
    leaf = jax.random.normal(key, (8, 5)) * jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 0.01, 3.0])[:, None]
    t = {"w": leaf}
    C = 1.0
    clipped = per_sample_clip(t, C)
    norms = per_sample_norms(clipped)
    assert jnp.all(norms <= C + 1e-5)


def test_per_sample_clip_preserves_below_threshold():
    leaf = jnp.array([[0.1, 0.1], [0.05, 0.0], [0.2, 0.1]], dtype=jnp.float32)
    t = {"w": leaf}
    C = 10.0
    clipped = per_sample_clip(t, C)
    assert jnp.allclose(clipped["w"], leaf, rtol=1e-4)


def test_per_sample_apply_mask_zeros_masked():
    leaf = jnp.ones((4, 3), dtype=jnp.float32) * jnp.array([1.0, 2.0, 3.0, 4.0])[:, None]
    t = {"w": leaf}
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    out = per_sample_apply_mask(t, mask)
    assert jnp.all(out["w"][1] == 0.0)
    assert jnp.all(out["w"][3] == 0.0)
    assert jnp.allclose(out["w"][0], leaf[0])
    assert jnp.allclose(out["w"][2], leaf[2])
