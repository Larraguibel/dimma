"""Tests for the projection-mechanism noise-scale calibration.

External behaviour only: hand-computed formula values, monotonicity, the
``s``-independence of the Gaussian scale, and input validation.
"""

import math

import pytest

from dimma.accounting import laplace_noise_scale, gaussian_noise_scale


# --- Laplace scale: 2 L sqrt(s) / (n eps) -------------------------------------


def test_laplace_scale_hand_computed():
    # L=1, s=4, n=1000, eps=2 -> 2*1*2 / (1000*2) = 4/2000 = 0.002
    assert laplace_noise_scale(1.0, 4.0, 1000.0, 2.0) == pytest.approx(0.002)


def test_laplace_scale_matches_formula():
    L, s, n, eps = 2.5, 9.0, 512.0, 0.7
    expected = 2.0 * L * math.sqrt(s) / (n * eps)
    assert laplace_noise_scale(L, s, n, eps) == pytest.approx(expected, rel=1e-12)


def test_laplace_scale_decreases_in_epsilon():
    a = laplace_noise_scale(1.0, 4.0, 1000.0, 1.0)
    b = laplace_noise_scale(1.0, 4.0, 1000.0, 2.0)
    assert b < a


def test_laplace_scale_decreases_in_n():
    a = laplace_noise_scale(1.0, 4.0, 500.0, 1.0)
    b = laplace_noise_scale(1.0, 4.0, 1000.0, 1.0)
    assert b < a


def test_laplace_scale_increases_in_s():
    a = laplace_noise_scale(1.0, 4.0, 1000.0, 1.0)
    b = laplace_noise_scale(1.0, 16.0, 1000.0, 1.0)
    # sqrt(16)/sqrt(4) = 2 -> exactly doubles
    assert b == pytest.approx(2.0 * a, rel=1e-12)


# --- Gaussian scale: sqrt(8 ln(1.25/delta)) L / (n eps) -----------------------


def test_gaussian_scale_hand_computed():
    # L=1, n=1, eps=1, delta=1.25 -> ln(1.25/1.25)=0 -> scale 0
    assert gaussian_noise_scale(1.0, 1.0, 1.0, 1.25) == pytest.approx(0.0)


def test_gaussian_scale_matches_formula():
    L, n, eps, delta = 1.0, 1000.0, 2.0, 1e-5
    expected = math.sqrt(8.0 * math.log(1.25 / delta)) * L / (n * eps)
    assert gaussian_noise_scale(L, n, eps, delta) == pytest.approx(
        expected, rel=1e-12
    )


def test_gaussian_scale_decreases_in_epsilon():
    a = gaussian_noise_scale(1.0, 1000.0, 1.0, 1e-5)
    b = gaussian_noise_scale(1.0, 1000.0, 2.0, 1e-5)
    assert b < a


def test_gaussian_scale_decreases_in_n():
    a = gaussian_noise_scale(1.0, 500.0, 1.0, 1e-5)
    b = gaussian_noise_scale(1.0, 1000.0, 1.0, 1e-5)
    assert b < a


def test_gaussian_scale_grows_as_delta_shrinks():
    # smaller delta -> larger ln(1.25/delta) -> larger scale
    a = gaussian_noise_scale(1.0, 1000.0, 1.0, 1e-3)
    b = gaussian_noise_scale(1.0, 1000.0, 1.0, 1e-7)
    assert b > a


def test_gaussian_scale_is_s_independent():
    # gaussian_noise_scale has no s argument; confirm the accounting layer's
    # Gaussian calibration truly does not depend on sparsity. Same (L,n,eps,
    # delta) must give one value regardless of any s a caller might imagine.
    val = gaussian_noise_scale(1.0, 1000.0, 1.0, 1e-5)
    # Reconstruct by hand with l_2 sensitivity 2L/n (no sqrt(s) anywhere).
    delta = 1e-5
    sensitivity_l2 = 2.0 * 1.0 / 1000.0
    expected = math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity_l2 / 1.0
    assert val == pytest.approx(expected, rel=1e-12)


# --- Validation ---------------------------------------------------------------
# These are pure closed-form functions and do not themselves validate; the
# eager validation lives in projection_mechanism. Here we simply confirm the
# scales are finite and positive on valid inputs (guards against sign/typo).


def test_scales_positive_on_valid_inputs():
    assert laplace_noise_scale(1.0, 4.0, 1000.0, 1.0) > 0.0
    assert gaussian_noise_scale(1.0, 1000.0, 1.0, 1e-5) > 0.0
