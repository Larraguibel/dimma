import pytest

from dimma.accounting import (
    poisson_gaussian_epsilon,
    poisson_gaussian_truncated_epsilon,
)


def _skip_if_no_accountant(val):
    if val is None:
        pytest.skip("dp_accounting not available")


def test_returns_positive_float():
    eps = poisson_gaussian_epsilon(0.01, 1.0, 1000, 1e-5)
    _skip_if_no_accountant(eps)
    assert isinstance(eps, float)
    assert eps > 0


def test_monotonic_in_compositions():
    eps_list = [poisson_gaussian_epsilon(0.01, 1.0, k, 1e-5)
                for k in (100, 1000, 10000)]
    for v in eps_list:
        _skip_if_no_accountant(v)
    assert eps_list[0] < eps_list[1] < eps_list[2]


def test_monotonic_in_noise_multiplier():
    eps_list = [poisson_gaussian_epsilon(0.01, z, 1000, 1e-5)
                for z in (0.5, 1.0, 2.0)]
    for v in eps_list:
        _skip_if_no_accountant(v)
    assert eps_list[0] > eps_list[1] > eps_list[2]


# This equality reflects Phase 3a's heuristic — see the
# poisson_gaussian_truncated_epsilon docstring. Will diverge once a
# tighter bound is implemented.
@pytest.mark.parametrize("args", [
    (0.01, 1.0, 1000, 1e-5),
    (0.05, 0.7, 500, 1e-6),
    (0.001, 2.5, 10000, 1e-5),
])
def test_truncated_equals_strict_currently(args):
    a = poisson_gaussian_epsilon(*args)
    b = poisson_gaussian_truncated_epsilon(*args)
    if a is None and b is None:
        pytest.skip("dp_accounting not available")
    assert a == b


def test_unavailable_dp_accounting():
    pytest.skip("requires controlled dp_accounting unavailability")
