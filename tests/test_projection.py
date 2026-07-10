"""External-behavior tests for the l1-ball projection primitives.

Correctness is checked against an *independent* NumPy reference that uses a
different algorithm (bisection on the soft-threshold) rather than a copy of
the sort-based implementation under test.

These tests run in JAX's default float32. Tolerances are loosened where
float32 cumulative-sum precision matters; the exact-identity assertions
(``==``) rely only on the ``jnp.where`` short-circuit, not on arithmetic, so
they are exact regardless of dtype.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dimma.core import project_l1_ball, project_l1_ball_pytree

# float32 rounding of the sort/cumsum/soft-threshold path.
ATOL = 1e-4
RTOL = 1e-4


# ---------------------------------------------------------------------------
# Independent NumPy reference: bisection on the soft-threshold theta.
# ---------------------------------------------------------------------------
def _project_l1_reference(x, radius):
    """Project x onto {z : ||z||_1 <= radius} by bisecting theta.

    Independent of the implementation under test: instead of sorting and a
    closed-form theta, we binary-search the threshold theta >= 0 for which
    sum(max(|x| - theta, 0)) == radius, then soft-threshold.
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    if a.sum() <= radius:
        return x.copy()
    lo, hi = 0.0, a.max()
    for _ in range(200):
        theta = 0.5 * (lo + hi)
        if np.maximum(a - theta, 0.0).sum() > radius:
            lo = theta
        else:
            hi = theta
    theta = 0.5 * (lo + hi)
    return np.sign(x) * np.maximum(a - theta, 0.0)


def _l1(v):
    return float(jnp.sum(jnp.abs(v)))


# ---------------------------------------------------------------------------
# Correctness vs the independent reference
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_matches_numpy_reference(seed):
    rng = np.random.default_rng(seed)
    d = int(rng.integers(1, 200))
    x = (rng.standard_normal(d) * rng.uniform(0.1, 5.0)).astype(np.float32)
    radius = float(rng.uniform(0.1, 3.0))

    got = np.asarray(project_l1_ball(jnp.asarray(x), radius))
    ref = _project_l1_reference(x, radius)
    np.testing.assert_allclose(got, ref, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Feasibility: ||P(x)||_1 <= radius
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(8))
def test_feasibility(seed):
    rng = np.random.default_rng(1000 + seed)
    d = int(rng.integers(1, 300))
    x = jnp.asarray((rng.standard_normal(d) * 10.0).astype(np.float32))
    radius = float(rng.uniform(0.05, 2.0))
    p = project_l1_ball(x, radius)
    # float32 cumsum can overshoot slightly; allow a small relative slack.
    assert _l1(p) <= radius * (1.0 + 1e-4) + 1e-4


# ---------------------------------------------------------------------------
# Exact idempotence: a point already inside the ball is returned bit-exactly
# ---------------------------------------------------------------------------
def test_inside_ball_returned_bitexact():
    x = jnp.array([0.1, -0.2, 0.05, 0.0, -0.15])  # ||x||_1 = 0.5
    p = project_l1_ball(x, 10.0)
    assert bool(jnp.all(p == x))


def test_projected_point_is_idempotent():
    rng = np.random.default_rng(7)
    x = jnp.asarray((rng.standard_normal(50) * 5.0).astype(np.float32))
    radius = 1.3
    p1 = project_l1_ball(x, radius)
    p2 = project_l1_ball(p1, radius)
    # p1 is (numerically) on/inside the ball, so projecting again is ~a no-op.
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p2), atol=ATOL)


# ---------------------------------------------------------------------------
# Contraction: projection does not increase l2 distance to any point in K
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_contraction_toward_point_in_ball(seed):
    rng = np.random.default_rng(2000 + seed)
    d = int(rng.integers(2, 100))
    radius = float(rng.uniform(0.5, 2.0))
    x = jnp.asarray((rng.standard_normal(d) * 8.0).astype(np.float32))  # outside

    # An arbitrary point strictly inside the ball.
    y = rng.standard_normal(d)
    y = (y / np.abs(y).sum() * (0.5 * radius)).astype(np.float32)
    yj = jnp.asarray(y)
    assert _l1(yj) <= radius + 1e-5

    p = project_l1_ball(x, radius)
    d_before = float(jnp.linalg.norm(x - yj))
    d_after = float(jnp.linalg.norm(p - yj))
    assert d_after <= d_before + 1e-4


# ---------------------------------------------------------------------------
# Pytree projection == flat projection on the raveled pytree
# ---------------------------------------------------------------------------
def test_pytree_equals_flat_on_raveled():
    from jax.flatten_util import ravel_pytree

    tree = {
        "w": jnp.array([[1.0, -2.0], [3.0, 0.5]]),
        "b": jnp.array([-4.0, 0.25, 2.0]),
    }
    radius = 2.5
    flat, unravel = ravel_pytree(tree)
    expected = unravel(project_l1_ball(flat, radius))
    got = project_l1_ball_pytree(tree, radius)

    for k in tree:
        np.testing.assert_array_equal(np.asarray(got[k]), np.asarray(expected[k]))
    # And the global l1 constraint holds across all leaves.
    total_l1 = sum(_l1(v) for v in jax.tree_util.tree_leaves(got))
    assert total_l1 <= radius * (1.0 + 1e-4) + 1e-4


# ---------------------------------------------------------------------------
# jit with a traced radius == eager result
# ---------------------------------------------------------------------------
def test_jit_traced_radius_matches_eager():
    rng = np.random.default_rng(11)
    x = jnp.asarray((rng.standard_normal(120) * 4.0).astype(np.float32))
    radius = 1.7

    jitted = jax.jit(project_l1_ball)
    eager = project_l1_ball(x, radius)
    traced = jitted(x, jnp.asarray(radius))  # radius as a traced array
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(traced))


def test_jit_pytree_traced_radius():
    tree = {"a": jnp.array([5.0, -3.0, 2.0]), "b": jnp.array([[1.0, -7.0]])}
    radius = jnp.asarray(1.1)
    jitted = jax.jit(project_l1_ball_pytree)
    out = jitted(tree, radius)
    total_l1 = sum(_l1(v) for v in jax.tree_util.tree_leaves(out))
    assert total_l1 <= float(radius) * (1.0 + 1e-4) + 1e-4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_d1_inside():
    x = jnp.array([0.3])
    p = project_l1_ball(x, 1.0)
    assert bool(jnp.all(p == x))


def test_d1_outside():
    x = jnp.array([5.0])
    p = project_l1_ball(x, 1.0)
    np.testing.assert_allclose(np.asarray(p), np.array([1.0]), atol=ATOL)

    xn = jnp.array([-5.0])
    pn = project_l1_ball(xn, 1.0)
    np.testing.assert_allclose(np.asarray(pn), np.array([-1.0]), atol=ATOL)


def test_zero_vector():
    x = jnp.zeros(10)
    p = project_l1_ball(x, 1.0)
    assert bool(jnp.all(p == x))
    # Also robust with radius 0 (rho guard prevents divide-by-zero).
    p0 = project_l1_ball(x, 0.0)
    assert _l1(p0) <= 1e-12


def test_exactly_on_boundary():
    # Values exactly representable in float32 so ||x||_1 == 1.0 exactly.
    x = jnp.array([0.5, -0.25, 0.25])  # ||x||_1 = 1.0
    p = project_l1_ball(x, 1.0)
    # On the boundary => already feasible => returned unchanged.
    assert bool(jnp.all(p == x))
    assert abs(_l1(p) - 1.0) < 1e-6
