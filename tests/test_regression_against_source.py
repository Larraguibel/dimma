"""Regression tests pinning dimma.core primitives to the source impl.

These tests must pass for Phase 1 to be considered complete. They are
the concrete check that the extraction did not silently alter any DP
primitive's numerical behavior.

If these tests fail, the extraction is broken and must be fixed before
proceeding to Phase 2 (which depends on Phase 1's correctness).
"""

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).parent.parent.parent / "private_spider_boost_criteo"
sys.path.insert(0, str(SOURCE_ROOT))

import jax
import jax.numpy as jnp
import pytest

import src.private_spiderboost as source_psb
import src.device as source_device
import src.privacy_accountant as source_acc

from dimma.algorithms.spiderboost import (
    make_anchor_step,
    make_variation_step,
    sgd_update,
)
from dimma.accounting import compute_noise_scales, verify_epsilon

from dimma.core import (
    pytree_global_norm,
    pytree_sub,
    pytree_add,
    pytree_scale,
    pytree_zeros_like,
    pytree_sum_over_batch,
    per_sample_norms,
    per_sample_clip,
    per_sample_apply_mask,
    add_pytree_gaussian_noise,
)
from dimma.utils import resolve_device


def _rand_pytree(key, shapes):
    keys = jax.random.split(key, len(shapes))
    return {f"l{i}": jax.random.normal(k, s) for i, (k, s) in enumerate(zip(keys, shapes))}


def _assert_tree_equal(a, b):
    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        assert jnp.array_equal(x, y)


def test_pytree_global_norm_regression():
    t = _rand_pytree(jax.random.PRNGKey(0), [(4,), (3, 5), (2, 2, 2)])
    assert jnp.array_equal(pytree_global_norm(t), source_psb.pytree_global_norm(t))


def test_pytree_sub_regression():
    a = _rand_pytree(jax.random.PRNGKey(1), [(4,), (3, 5)])
    b = _rand_pytree(jax.random.PRNGKey(2), [(4,), (3, 5)])
    _assert_tree_equal(pytree_sub(a, b), source_psb.pytree_sub(a, b))


def test_pytree_add_regression():
    a = _rand_pytree(jax.random.PRNGKey(1), [(4,), (3, 5)])
    b = _rand_pytree(jax.random.PRNGKey(2), [(4,), (3, 5)])
    _assert_tree_equal(pytree_add(a, b), source_psb.pytree_add(a, b))


def test_pytree_scale_regression():
    t = _rand_pytree(jax.random.PRNGKey(3), [(4,), (3, 5)])
    _assert_tree_equal(pytree_scale(t, 0.37), source_psb.pytree_scale(t, 0.37))


def test_pytree_zeros_like_regression():
    t = _rand_pytree(jax.random.PRNGKey(4), [(4,), (3, 5)])
    _assert_tree_equal(pytree_zeros_like(t), source_psb.pytree_zeros_like(t))


def test_pytree_sum_over_batch_regression():
    t = _rand_pytree(jax.random.PRNGKey(5), [(8, 3), (8, 2, 4)])
    _assert_tree_equal(pytree_sum_over_batch(t), source_psb.pytree_sum_over_batch(t))


def test_per_sample_norms_regression():
    t = _rand_pytree(jax.random.PRNGKey(6), [(16, 3), (16, 2, 4)])
    assert jnp.array_equal(per_sample_norms(t), source_psb.per_sample_norms(t))


def test_per_sample_clip_regression():
    t = _rand_pytree(jax.random.PRNGKey(7), [(16, 3), (16, 2, 4)])
    _assert_tree_equal(per_sample_clip(t, 1.0), source_psb.per_sample_clip(t, 1.0))


def test_per_sample_apply_mask_regression():
    t = _rand_pytree(jax.random.PRNGKey(8), [(16, 3), (16, 2, 4)])
    mask = (jax.random.uniform(jax.random.PRNGKey(9), (16,)) > 0.5).astype(jnp.float32)
    _assert_tree_equal(
        per_sample_apply_mask(t, mask),
        source_psb.per_sample_apply_mask(t, mask),
    )


def test_add_pytree_gaussian_noise_regression():
    t = _rand_pytree(jax.random.PRNGKey(10), [(4,), (3, 5)])
    key = jax.random.PRNGKey(42)
    _assert_tree_equal(
        add_pytree_gaussian_noise(t, key, 0.5),
        source_psb.add_pytree_gaussian_noise(t, key, 0.5),
    )


def test_resolve_device_regression():
    assert resolve_device("cpu") == source_device.resolve_device("cpu")


# ---------------------------------------------------------------------------
# Phase 2 regressions: kernels and accountant
# ---------------------------------------------------------------------------


def _per_sample_loss_reg(w, x, y):
    return 0.5 * (jnp.dot(w, x) - y) ** 2


def _grad_fn_reg():
    return jax.vmap(jax.grad(_per_sample_loss_reg), in_axes=(None, 0, 0))


def test_anchor_step_regression():
    grad_fn = _grad_fn_reg()
    dimma_step = make_anchor_step(grad_fn)
    source_step = source_psb.make_anchor_step(grad_fn)

    B = 8
    w = jnp.array([0.3, -0.7])
    x = jax.random.normal(jax.random.PRNGKey(100), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(101), (B,))
    mask = (jax.random.uniform(jax.random.PRNGKey(102), (B,)) > 0.3).astype(jnp.float32)
    key = jax.random.PRNGKey(103)

    o_d = dimma_step(w, x, y, mask, B, 1.0, 0.5, key)
    o_s = source_step(w, x, y, mask, B, 1.0, 0.5, key)
    assert jnp.array_equal(o_d.grad_estimate, o_s.grad_estimate)
    assert jnp.array_equal(o_d.grad_norm, o_s.grad_norm)


def test_variation_step_regression():
    grad_fn = _grad_fn_reg()
    dimma_step = make_variation_step(grad_fn)
    source_step = source_psb.make_variation_step(grad_fn)

    B = 8
    w_t = jnp.array([0.5, -0.3])
    w_prev = jnp.array([0.4, -0.25])
    prev_grad = jnp.array([0.1, 0.2])
    x = jax.random.normal(jax.random.PRNGKey(200), (B, 2))
    y = jax.random.normal(jax.random.PRNGKey(201), (B,))
    mask = (jax.random.uniform(jax.random.PRNGKey(202), (B,)) > 0.3).astype(jnp.float32)
    key = jax.random.PRNGKey(203)

    o_d = dimma_step(w_t, w_prev, prev_grad, x, y, mask, B, 1.0, 0.5, 1.0, key)
    o_s = source_step(w_t, w_prev, prev_grad, x, y, mask, B, 1.0, 0.5, 1.0, key)
    assert jnp.array_equal(o_d.grad_estimate, o_s.grad_estimate)
    assert jnp.array_equal(o_d.grad_norm, o_s.grad_norm)


def test_sgd_update_regression():
    params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}
    grad = {"w": jnp.array([0.3, -0.1]), "b": jnp.array([0.05])}
    _assert_tree_equal(
        sgd_update(params, grad, 0.07),
        source_psb.sgd_update(params, grad, 0.07),
    )


_NS_GRID = [
    # (L0, L1, eps, delta, T, q, n, b1, b2, c)
    (1.0, 1.0, 1.0, 1e-5, 1000, 50, 10000, 512, 64, 1.0),
    (0.5, 2.0, 0.5, 1e-6, 100, 1, 5000, 256, 32, 1.0),
    (1.0, 1.0, 2.0, 1e-5, 100, 100, 10000, 128, 128, 1.5),
    (2.0, 0.5, 1.0, 1e-4, 500, 10, 20000, 1024, 256, 1.0),
]


@pytest.mark.parametrize("args", _NS_GRID)
def test_compute_noise_scales_regression(args):
    d = compute_noise_scales(*args)
    s = source_acc.compute_noise_scales(*args)
    assert abs(d.sigma1 - s.sigma1) < 1e-12
    assert abs(d.sigma2 - s.sigma2) < 1e-12
    assert abs(d.sigma2_hat - s.sigma2_hat) < 1e-12


@pytest.mark.parametrize("args", _NS_GRID)
def test_verify_epsilon_regression(args):
    L0, L1, eps, delta, T, q, n, b1, b2, c = args
    ns = compute_noise_scales(*args)
    d = verify_epsilon(L0, delta, T, q, n, b1, b2, ns.sigma1, ns.sigma2_hat)
    s = source_acc.verify_epsilon(L0, eps, delta, T, q, n, b1, b2, ns.sigma1, ns.sigma2_hat)
    if d is None and s is None:
        return
    assert d is not None and s is not None
    assert abs(d - s) < 1e-10
