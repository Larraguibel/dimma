import jax
import jax.numpy as jnp
import pytest

from dimma.algorithms.spiderboost import (
    make_anchor_step,
    make_variation_step,
    sgd_update,
)
from dimma.core import pytree_global_norm


def _per_sample_loss(w, x, y):
    return 0.5 * (jnp.dot(w, x) - y) ** 2


def _make_grad_fn():
    return jax.vmap(jax.grad(_per_sample_loss), in_axes=(None, 0, 0))


def test_sgd_update():
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    grad = {"w": jnp.array([0.5, 0.5, 0.5])}
    out = sgd_update(params, grad, lr=0.1)
    expected = jnp.array([0.95, 1.95, 2.95])
    assert jnp.allclose(out["w"], expected, atol=1e-6)


def test_anchor_step_no_noise_no_clip():
    grad_fn = _make_grad_fn()
    anchor = make_anchor_step(grad_fn)
    key = jax.random.PRNGKey(0)
    B = 4
    w = jnp.array([0.5, -0.3])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    out = anchor(w, x, y, mask, b1=B, L0=1e10, sigma1=0.0, key=key)

    per_sample = grad_fn(w, x, y)
    expected = jnp.mean(per_sample, axis=0)
    assert jnp.allclose(out.grad_estimate, expected, atol=1e-5)
    assert jnp.allclose(out.grad_norm, pytree_global_norm(out.grad_estimate))


def test_anchor_step_mask_divides_by_b1_not_count():
    grad_fn = _make_grad_fn()
    anchor = make_anchor_step(grad_fn)
    B = 4
    w = jnp.array([0.5, -0.3])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.array([1.0, 0.0, 1.0, 0.0])
    out = anchor(w, x, y, mask, b1=B, L0=1e10, sigma1=0.0, key=jax.random.PRNGKey(0))

    per_sample = grad_fn(w, x, y)
    expected = (per_sample[0] + per_sample[2]) / B
    assert jnp.allclose(out.grad_estimate, expected, atol=1e-5)


def test_anchor_step_clipping_active():
    grad_fn = _make_grad_fn()
    anchor = make_anchor_step(grad_fn)
    B = 4
    w = jnp.array([10.0, -10.0])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2)) * 5
    y = jnp.zeros(B)
    mask = jnp.ones(B)
    L0 = 1.0
    out = anchor(w, x, y, mask, b1=B, L0=L0, sigma1=0.0, key=jax.random.PRNGKey(0))

    from dimma.core.clipping import per_sample_clip
    per_sample = grad_fn(w, x, y)
    clipped = per_sample_clip(per_sample, L0)
    expected = jnp.mean(clipped, axis=0)
    assert jnp.allclose(out.grad_estimate, expected, atol=1e-5)


def test_variation_step_zero_delta_w_returns_prev_grad():
    grad_fn = _make_grad_fn()
    var_step = make_variation_step(grad_fn)
    B = 4
    w = jnp.array([0.5, -0.3])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    prev_grad = jnp.array([0.7, -0.4])

    out = var_step(w, w, prev_grad, x, y, mask, b2=B, L1=1.0,
                   sigma2=2.0, sigma2_hat=3.0, key=jax.random.PRNGKey(0))
    assert jnp.allclose(out.grad_estimate, prev_grad, atol=1e-5)


def test_variation_step_no_noise():
    grad_fn = _make_grad_fn()
    var_step = make_variation_step(grad_fn)
    B = 4
    w_t = jnp.array([0.5, -0.3])
    w_prev = jnp.array([0.4, -0.2])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    prev_grad = jnp.array([0.7, -0.4])
    L1 = 100.0  # large so clipping doesn't activate

    out = var_step(w_t, w_prev, prev_grad, x, y, mask, b2=B, L1=L1,
                   sigma2=0.0, sigma2_hat=0.0, key=jax.random.PRNGKey(0))

    from dimma.core.pytree import pytree_global_norm, pytree_sub
    from dimma.core.clipping import per_sample_clip
    delta_w = pytree_global_norm(pytree_sub(w_t, w_prev))
    clip_c = L1 * delta_w
    diff = grad_fn(w_t, x, y) - grad_fn(w_prev, x, y)
    clipped = per_sample_clip(diff, clip_c)
    expected = prev_grad + jnp.sum(clipped, axis=0) / B
    assert jnp.allclose(out.grad_estimate, expected, atol=1e-5)
