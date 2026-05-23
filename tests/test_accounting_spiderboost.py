import math
import pytest

from dimma.accounting import (
    DEFAULT_C,
    NoiseScales,
    compute_noise_scales,
    verify_epsilon,
)


def test_compute_noise_scales_matches_formula():
    L0, L1, eps, delta = 1.0, 1.0, 1.0, 1e-5
    T, q, n, b1, b2, c = 1000, 50, 10000, 512, 64, 1.0
    out = compute_noise_scales(L0, L1, eps, delta, T, q, n, b1, b2, c)

    log_inv_delta = math.log(1.0 / delta)
    sqrt_T = math.sqrt(T)
    factor_anchor = max(1.0 / b1, sqrt_T / (q * n))
    factor_var = max(1.0 / b2, sqrt_T / n)
    base = c * math.sqrt(log_inv_delta) / eps
    s1 = base * L0 * factor_anchor
    s2 = base * L1 * factor_var
    s2h = base * 2.0 * L0 * factor_var

    assert abs(out.sigma1 - s1) < 1e-9
    assert abs(out.sigma2 - s2) < 1e-9
    assert abs(out.sigma2_hat - s2h) < 1e-9


def test_noise_scales_field_ordering():
    a = NoiseScales(1.0, 2.0, 3.0)
    b = NoiseScales(sigma1=1.0, sigma2=2.0, sigma2_hat=3.0)
    assert a == b
    assert a.sigma1 == 1.0
    assert a.sigma2 == 2.0
    assert a.sigma2_hat == 3.0


def test_verify_epsilon_returns_float_or_none():
    out = verify_epsilon(
        L0=1.0, delta=1e-5,
        T=100, q=10, n=10000, b1=512, b2=64,
        sigma1=1.0, sigma2_hat=2.0,
    )
    if out is None:
        return
    assert isinstance(out, float)
    assert out > 0


def test_verify_epsilon_mechanism_counts():
    """Pins current T+1 semantics: anchor=ceil(T/q)+1, var=T+1-anchor.

    Phase 3 will replace these expected counts with (ceil(T/q), T - ceil(T/q)).
    """
    try:
        import dp_accounting  # noqa: F401
    except ImportError:
        pytest.skip("dp_accounting not available")

    from dimma.accounting import spiderboost as acc_mod
    from dp_accounting import rdp

    calls = []
    real_compose = rdp.RdpAccountant.compose

    def recording_compose(self, event, count=1):
        calls.append(count)
        return real_compose(self, event, count=count)

    rdp.RdpAccountant.compose = recording_compose
    try:
        verify_epsilon(
            L0=1.0, delta=1e-5,
            T=100, q=10, n=10000, b1=512, b2=64,
            sigma1=1.0, sigma2_hat=2.0,
        )
    finally:
        rdp.RdpAccountant.compose = real_compose

    assert calls == [11, 90]


def test_default_c_is_one():
    assert DEFAULT_C == 1.0
