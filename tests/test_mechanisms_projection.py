"""Tests for the standalone projection mechanism (Ghazi et al. 2024, Alg. 1).

Covers branch-dispatch equivalence to a manual reconstruction, output
feasibility, the Lemma 3.1 deterministic error bound over many trials for both
noise branches, a dimension-independence smoke test, and input validation.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.accounting import laplace_noise_scale, gaussian_noise_scale
from dimma.core import (
    add_pytree_gaussian_noise,
    add_pytree_laplace_noise,
    project_l1_ball_pytree,
)
from dimma.mechanisms import projection_mechanism


# --- helpers ------------------------------------------------------------------


def _sparse_mean(rng, d, s, L=1.0):
    """An ``s``-sparse vector with ``l_2 <= L`` (hence inside K = B_1(0, L√s))."""
    x = np.zeros(d, dtype=np.float32)
    idx = rng.choice(d, size=s, replace=False)
    vals = rng.standard_normal(s).astype(np.float32)
    x[idx] = vals
    norm = np.linalg.norm(x)
    if norm > 0:
        # scale to l_2 = 0.8 * L so the mean sits strictly inside the ball.
        x = x * (0.8 * L / norm)
    return jnp.asarray(x)


def _l1(a):
    return float(jnp.sum(jnp.abs(a)))


def _l2(a):
    return float(jnp.linalg.norm(a))


# --- branch dispatch is bit-exact vs manual reconstruction --------------------


def test_laplace_branch_bit_exact_reconstruction():
    L, s, n, eps = 1.0, 5.0, 1000.0, 1.0
    d = 200
    rng = np.random.default_rng(0)
    mean = _sparse_mean(rng, d, int(s), L)
    key = jax.random.PRNGKey(123)

    zhat = projection_mechanism(
        mean, epsilon=eps, delta=0.0, n=n, L=L, s=s, key=key
    )

    scale = laplace_noise_scale(L, s, n, eps)
    z_tilde_manual = add_pytree_laplace_noise(mean, key, scale)
    zhat_manual = project_l1_ball_pytree(z_tilde_manual, L * math.sqrt(s))

    assert jnp.array_equal(zhat, zhat_manual)


def test_gaussian_branch_bit_exact_reconstruction():
    L, s, n, eps, delta = 1.0, 5.0, 1000.0, 1.0, 1e-5
    d = 200
    rng = np.random.default_rng(1)
    mean = _sparse_mean(rng, d, int(s), L)
    key = jax.random.PRNGKey(456)

    zhat = projection_mechanism(
        mean, epsilon=eps, delta=delta, n=n, L=L, s=s, key=key
    )

    std = gaussian_noise_scale(L, n, eps, delta)
    z_tilde_manual = add_pytree_gaussian_noise(mean, key, std)
    zhat_manual = project_l1_ball_pytree(z_tilde_manual, L * math.sqrt(s))

    assert jnp.array_equal(zhat, zhat_manual)


def test_return_noisy_convention():
    L, s, n, eps, delta = 1.0, 5.0, 1000.0, 1.0, 1e-5
    rng = np.random.default_rng(2)
    mean = _sparse_mean(rng, 128, int(s), L)
    key = jax.random.PRNGKey(7)

    zhat_only = projection_mechanism(
        mean, epsilon=eps, delta=delta, n=n, L=L, s=s, key=key
    )
    zhat, z_tilde = projection_mechanism(
        mean, epsilon=eps, delta=delta, n=n, L=L, s=s, key=key,
        return_noisy=True,
    )
    # return_noisy must not perturb the projected output.
    assert jnp.array_equal(zhat, zhat_only)
    # z_tilde is the pre-projection noisy vector: same shape as the mean, and
    # projecting it reproduces zhat.
    assert z_tilde.shape == mean.shape
    assert jnp.array_equal(
        project_l1_ball_pytree(z_tilde, L * math.sqrt(s)), zhat
    )


def test_pytree_mean_single_code_path():
    """A nested pytree mean is projected globally, same as its flat raveling."""
    L, s, n, eps, delta = 1.0, 6.0, 1000.0, 1.0, 1e-5
    key = jax.random.PRNGKey(99)
    pytree = {"w": jnp.array([0.1, 0.0, -0.2, 0.0]), "b": jnp.array([0.05, 0.0])}

    zhat = projection_mechanism(
        pytree, epsilon=eps, delta=delta, n=n, L=L, s=s, key=key
    )
    flat = jnp.concatenate([zhat["w"], zhat["b"]])
    assert _l1(flat) <= L * math.sqrt(s) + 1e-4
    assert set(zhat.keys()) == {"w", "b"}


# --- output feasibility -------------------------------------------------------


@pytest.mark.parametrize("delta", [0.0, 1e-5])
def test_output_feasible_in_l1_ball(delta):
    L, s, n, eps = 1.0, 5.0, 200.0, 1.0
    d = 500
    rng = np.random.default_rng(3)
    radius = L * math.sqrt(s)
    for t in range(20):
        mean = _sparse_mean(rng, d, int(s), L)
        key = jax.random.PRNGKey(t)
        zhat = projection_mechanism(
            mean, epsilon=eps, delta=delta, n=n, L=L, s=s, key=key
        )
        assert _l1(zhat) <= radius + 1e-4


# --- Lemma 3.1 deterministic per-trial bound ----------------------------------


@pytest.mark.parametrize("delta", [0.0, 1e-5])
def test_lemma_31_bound_per_trial(delta):
    # ‖ẑ − z̄‖_2 <= sqrt(2 L ‖ξ‖_∞ sqrt(s)) almost surely (both branches),
    # provided z̄ ∈ K (guaranteed: mean is s-sparse with l_2 < L).
    L, s, n, eps = 1.0, 5.0, 100.0, 1.0
    d = 300
    rng = np.random.default_rng(4)
    for t in range(100):
        mean = _sparse_mean(rng, d, int(s), L)
        key = jax.random.PRNGKey(1000 + t)
        zhat, z_tilde = projection_mechanism(
            mean, epsilon=eps, delta=delta, n=n, L=L, s=s, key=key,
            return_noisy=True,
        )
        xi = z_tilde - mean
        xi_inf = float(jnp.max(jnp.abs(xi)))
        lhs = _l2(zhat - mean)
        rhs = math.sqrt(2.0 * L * xi_inf * math.sqrt(s))
        # small absolute slack for float32 accumulation in the projection.
        assert lhs <= rhs + 1e-4


# --- dimension independence ---------------------------------------------------


def test_dimension_independence_smoke():
    # Projected l_2 error stays ~flat as d grows, while the unprojected noise
    # l_2 error grows ~sqrt(d). Fixed (L, s, n, eps, delta) -> fixed noise
    # scale, so only d changes the picture.
    L, s, n, eps, delta = 1.0, 5.0, 1000.0, 1.0, 1e-5
    dims = [100, 10000]
    n_trials = 30

    proj_err = {}
    unproj_err = {}
    for d in dims:
        rng = np.random.default_rng(10 * d)
        pe, ue = [], []
        for t in range(n_trials):
            mean = _sparse_mean(rng, d, int(s), L)
            key = jax.random.PRNGKey(t)
            zhat, z_tilde = projection_mechanism(
                mean, epsilon=eps, delta=delta, n=n, L=L, s=s, key=key,
                return_noisy=True,
            )
            pe.append(_l2(zhat - mean))
            ue.append(_l2(z_tilde - mean))
        proj_err[d] = float(np.mean(pe))
        unproj_err[d] = float(np.mean(ue))

    dim_ratio = dims[1] / dims[0]  # 100
    sqrt_ratio = math.sqrt(dim_ratio)  # 10

    # Unprojected error grows ~sqrt(d): the ratio should be close to sqrt_ratio
    # (loose band).
    unproj_ratio = unproj_err[dims[1]] / unproj_err[dims[0]]
    assert 0.5 * sqrt_ratio < unproj_ratio < 2.0 * sqrt_ratio

    # Projected error is nearly dimension-independent: it must grow far more
    # slowly than the unprojected error (well below the sqrt(d) growth).
    proj_ratio = proj_err[dims[1]] / proj_err[dims[0]]
    assert proj_ratio < 0.5 * sqrt_ratio
    # And at the largest dimension, projection is a big win in absolute terms.
    assert proj_err[dims[1]] < unproj_err[dims[1]]


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(epsilon=0.0, delta=1e-5, n=100, L=1.0, s=5.0),
        dict(epsilon=-1.0, delta=1e-5, n=100, L=1.0, s=5.0),
        dict(epsilon=1.0, delta=-1e-5, n=100, L=1.0, s=5.0),
        dict(epsilon=1.0, delta=1e-5, n=0, L=1.0, s=5.0),
        dict(epsilon=1.0, delta=1e-5, n=100, L=1.0, s=0.5),
        dict(epsilon=1.0, delta=1e-5, n=100, L=0.0, s=5.0),
        dict(epsilon=1.0, delta=1e-5, n=100, L=-1.0, s=5.0),
    ],
)
def test_validation_raises(kwargs):
    mean = jnp.zeros(10)
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError):
        projection_mechanism(mean, key=key, **kwargs)


def test_valid_inputs_do_not_raise():
    mean = jnp.zeros(10)
    key = jax.random.PRNGKey(0)
    # delta = 0 is valid (pure-DP branch); s = 1, n = 1 are the boundaries.
    projection_mechanism(mean, epsilon=1.0, delta=0.0, n=1, L=1.0, s=1.0, key=key)
    projection_mechanism(
        mean, epsilon=1.0, delta=1e-5, n=1, L=1.0, s=1.0, key=key
    )
