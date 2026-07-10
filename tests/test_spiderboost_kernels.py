import math

import jax
import jax.numpy as jnp
import pytest
from jax.flatten_util import ravel_pytree

from dimma.algorithms.spiderboost import (
    make_anchor_step,
    make_variation_step,
    sgd_update,
)
from dimma.core import pytree_global_norm
from dimma.core.projection import project_l1_ball_pytree


def _l1_norm(tree):
    flat, _ = ravel_pytree(tree)
    return float(jnp.sum(jnp.abs(flat)))


def _assert_tree_equal(a, b):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert jnp.array_equal(x, y)


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


# ---------------------------------------------------------------------------
# Projection post-processing (Ghazi et al. 2024, Algorithm 1) — Phase 3.
# ---------------------------------------------------------------------------


def test_anchor_s_none_bit_identical_to_no_arg():
    grad_fn = _make_grad_fn()
    anchor_default = make_anchor_step(grad_fn)
    anchor_none = make_anchor_step(grad_fn, s=None)
    B = 4
    w = jnp.array([0.5, -0.3])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    key = jax.random.PRNGKey(7)
    out_a = anchor_default(w, x, y, mask, b1=B, L0=2.0, sigma1=1.0, key=key)
    out_b = anchor_none(w, x, y, mask, b1=B, L0=2.0, sigma1=1.0, key=key)
    _assert_tree_equal(out_a.grad_estimate, out_b.grad_estimate)
    assert jnp.array_equal(out_a.grad_norm, out_b.grad_norm)


def test_variation_s_none_bit_identical_to_no_arg():
    grad_fn = _make_grad_fn()
    var_default = make_variation_step(grad_fn)
    var_none = make_variation_step(grad_fn, s=None)
    B = 4
    w_t = jnp.array([0.5, -0.3])
    w_prev = jnp.array([0.4, -0.2])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    prev_grad = jnp.array([0.7, -0.4])
    key = jax.random.PRNGKey(7)
    out_a = var_default(w_t, w_prev, prev_grad, x, y, mask, b2=B, L1=1.0,
                        sigma2=2.0, sigma2_hat=3.0, key=key)
    out_b = var_none(w_t, w_prev, prev_grad, x, y, mask, b2=B, L1=1.0,
                     sigma2=2.0, sigma2_hat=3.0, key=key)
    _assert_tree_equal(out_a.grad_estimate, out_b.grad_estimate)
    assert jnp.array_equal(out_a.grad_norm, out_b.grad_norm)


def test_anchor_projection_enforces_l1_ball():
    grad_fn = _make_grad_fn()
    s = 2
    anchor = make_anchor_step(grad_fn, s=s)
    B = 4
    w = jnp.array([0.5, -0.3])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    L0 = 1.0
    # Large noise so the unprojected estimate escapes the ball, forcing a project.
    out = anchor(w, x, y, mask, b1=B, L0=L0, sigma1=5.0, key=jax.random.PRNGKey(11))
    radius = L0 * math.sqrt(s)
    assert _l1_norm(out.grad_estimate) <= radius + 1e-5


def test_variation_projection_zero_prev_enforces_l1_ball():
    grad_fn = _make_grad_fn()
    s = 2
    var_step = make_variation_step(grad_fn, s=s)
    B = 4
    w_t = jnp.array([0.5, -0.3])
    w_prev = jnp.array([0.4, -0.2])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    prev_grad = jnp.zeros(2)
    L1 = 1.0
    out = var_step(w_t, w_prev, prev_grad, x, y, mask, b2=B, L1=L1,
                   sigma2=50.0, sigma2_hat=50.0, key=jax.random.PRNGKey(11))
    delta_w = float(pytree_global_norm(w_t - w_prev))
    radius = L1 * delta_w * math.sqrt(2 * s)
    assert _l1_norm(out.grad_estimate) <= radius + 1e-5


# ---------------------------------------------------------------------------
# Eager validation of the sparsity hyperparameter s (issue #22).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_s", [0, -1, 2.5, True])
def test_anchor_step_rejects_invalid_s(bad_s):
    grad_fn = _make_grad_fn()
    with pytest.raises(ValueError) as exc:
        make_anchor_step(grad_fn, s=bad_s)
    assert "s must be a positive integer or None" in str(exc.value)


@pytest.mark.parametrize("bad_s", [0, -1, 2.5, True])
def test_variation_step_rejects_invalid_s(bad_s):
    grad_fn = _make_grad_fn()
    with pytest.raises(ValueError) as exc:
        make_variation_step(grad_fn, s=bad_s)
    assert "s must be a positive integer or None" in str(exc.value)


@pytest.mark.parametrize("good_s", [None, 1, 4])
def test_factories_accept_valid_s(good_s):
    # None and any positive int build a kernel without error.
    grad_fn = _make_grad_fn()
    assert callable(make_anchor_step(grad_fn, s=good_s))
    assert callable(make_variation_step(grad_fn, s=good_s))


def test_anchor_projection_hand_reconstruction():
    from dimma.core.pytree import (
        pytree_scale,
        pytree_sum_over_batch,
    )
    from dimma.core.clipping import per_sample_clip, per_sample_apply_mask
    from dimma.core.noise import add_pytree_gaussian_noise

    grad_fn = _make_grad_fn()
    s = 2
    anchor = make_anchor_step(grad_fn, s=s)
    B = 4
    w = jnp.array([0.5, -0.3])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    L0, sigma1 = 1.0, 5.0
    key = jax.random.PRNGKey(11)

    out = anchor(w, x, y, mask, b1=B, L0=L0, sigma1=sigma1, key=key)

    # Hand-reconstruct: project(noisy anchor estimate), same key, bit-exact.
    per_sample = per_sample_clip(grad_fn(w, x, y), L0)
    per_sample = per_sample_apply_mask(per_sample, mask)
    averaged = pytree_scale(pytree_sum_over_batch(per_sample), 1.0 / B)
    noisy = add_pytree_gaussian_noise(averaged, key, sigma1)
    expected = project_l1_ball_pytree(noisy, L0 * math.sqrt(s))

    _assert_tree_equal(out.grad_estimate, expected)


def test_variation_projection_nonzero_prev_hand_reconstruction():
    from dimma.core.pytree import (
        pytree_add,
        pytree_global_norm,
        pytree_scale,
        pytree_sub,
        pytree_sum_over_batch,
    )
    from dimma.core.clipping import per_sample_clip, per_sample_apply_mask
    from dimma.core.noise import add_pytree_gaussian_noise

    grad_fn = _make_grad_fn()
    s = 2
    var_step = make_variation_step(grad_fn, s=s)
    B = 4
    w_t = jnp.array([0.5, -0.3])
    w_prev = jnp.array([0.4, -0.2])
    x = jax.random.normal(jax.random.PRNGKey(1), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(2), (B,))
    mask = jnp.ones(B)
    prev_grad = jnp.array([0.7, -0.4])
    L1, sigma2, sigma2_hat = 1.0, 2.0, 3.0
    key = jax.random.PRNGKey(11)

    out = var_step(w_t, w_prev, prev_grad, x, y, mask, b2=B, L1=L1,
                   sigma2=sigma2, sigma2_hat=sigma2_hat, key=key)

    # Hand-reconstruct: prev + project(noisy_delta), same key, bit-exact.
    delta_w = pytree_global_norm(pytree_sub(w_t, w_prev))
    diff = pytree_sub(grad_fn(w_t, x, y), grad_fn(w_prev, x, y))
    diff = per_sample_clip(diff, L1 * delta_w)
    diff = per_sample_apply_mask(diff, mask)
    averaged = pytree_scale(pytree_sum_over_batch(diff), 1.0 / B)
    noise_std = jnp.minimum(sigma2 * delta_w, sigma2_hat)
    noisy_delta = add_pytree_gaussian_noise(averaged, key, noise_std)
    projected = project_l1_ball_pytree(noisy_delta, L1 * delta_w * math.sqrt(2 * s))
    expected = pytree_add(prev_grad, projected)

    _assert_tree_equal(out.grad_estimate, expected)
