"""Tests for ``resolve_config`` (Theorem B.3 hyperparameter resolver, issue #9).

Reference table (ADR-0002 / issue #7), at ``L0=1, F0=1``::

    n=800000, L1=5, T=200, d=3201, delta=1/n
        eps=0.1 -> q=29   (in-regime; returns a config)
        eps=1.0 -> q=2941 (q>T at T=200 -> raises, per ADR-0002 §5)
    eta = 1/(2*L1) = 0.1 always
"""

import math

import jax.numpy as jnp
import pytest

from dimma import resolve_config, TrainConfig


def _params_of_size(d: int):
    """A trivial pytree whose leaf sizes sum to ``d``."""
    return {"w": jnp.zeros((d,))}


# Reference-table inputs (L0=1, F0=1)
REF = dict(n=800000, F0=1.0, L0=1.0, L1=5.0, T=200, b1=800000, seed=0)
REF_DELTA = 1.0 / 800000
REF_D = 3201


# ---------------------------------------------------------------------------
# Derived values match the reference table
# ---------------------------------------------------------------------------

def test_reference_eps_0p1_derives_q29_eta_and_b2():
    cfg = resolve_config(
        _params_of_size(REF_D), REF["n"], REF["F0"],
        epsilon=0.1, delta=REF_DELTA,
        L0=REF["L0"], L1=REF["L1"], T=REF["T"], b1=REF["b1"], seed=REF["seed"],
    )
    assert isinstance(cfg, TrainConfig)
    assert cfg.q == 29
    assert cfg.eta == 1.0 / (2.0 * REF["L1"])  # 0.1
    assert cfg.b2 == 11589  # floor(max term)
    # caller-provided / passthrough fields land verbatim
    assert cfg.epsilon == 0.1
    assert cfg.delta == REF_DELTA
    assert cfg.L0 == REF["L0"] and cfg.L1 == REF["L1"]
    assert cfg.T == REF["T"] and cfg.b1 == REF["b1"] and cfg.seed == REF["seed"]


def test_reference_eps_1p0_q2941_surfaces_via_q_gt_T_guard():
    # At eps=1.0 the formula gives q=2941, which exceeds T=200 -> in-regime
    # guard fires (ADR-0002 §5). The reference value still surfaces in the
    # error message.
    with pytest.raises(ValueError) as exc:
        resolve_config(
            _params_of_size(REF_D), REF["n"], REF["F0"],
            epsilon=1.0, delta=REF_DELTA,
            L0=REF["L0"], L1=REF["L1"], T=REF["T"], b1=REF["b1"], seed=REF["seed"],
        )
    msg = str(exc.value)
    assert "2941" in msg
    assert "T=200" in msg


def test_eta_is_half_inverse_L1():
    cfg = resolve_config(
        _params_of_size(REF_D), REF["n"], REF["F0"],
        epsilon=0.1, delta=REF_DELTA,
        L0=REF["L0"], L1=2.0, T=REF["T"], b1=REF["b1"], seed=0,
    )
    assert cfg.eta == 0.25


# ---------------------------------------------------------------------------
# None ⇒ derive; explicit ⇒ pass through unchanged
# ---------------------------------------------------------------------------

def test_explicit_values_pass_through_unchanged():
    cfg = resolve_config(
        _params_of_size(REF_D), REF["n"], REF["F0"],
        epsilon=0.1, delta=REF_DELTA,
        L0=REF["L0"], L1=REF["L1"], T=REF["T"], b1=REF["b1"], seed=7,
        q=10, b2=100, eta=0.05,  # all explicit, in-regime
    )
    assert cfg.q == 10
    assert cfg.b2 == 100
    assert cfg.eta == 0.05


def test_partial_explicit_derives_only_the_rest():
    # provide q explicitly; eta and b2 derived
    cfg = resolve_config(
        _params_of_size(REF_D), REF["n"], REF["F0"],
        epsilon=0.1, delta=REF_DELTA,
        L0=REF["L0"], L1=REF["L1"], T=REF["T"], b1=REF["b1"], seed=0,
        q=15,
    )
    assert cfg.q == 15            # respected
    assert cfg.eta == 0.1         # derived
    assert cfg.b2 == 11589        # derived


def test_d_is_computed_from_pytree_leaves():
    # nested pytree, sizes 100 + 100 + 1 = 201; q scales with 1/d. Use a long
    # horizon so the smaller d (larger q) stays in-regime (q <= T).
    params = {"a": jnp.zeros((10, 10)), "b": jnp.zeros((100,)), "c": jnp.zeros(())}
    T = 2000
    cfg = resolve_config(
        params, REF["n"], REF["F0"],
        epsilon=0.1, delta=REF_DELTA,
        L0=REF["L0"], L1=REF["L1"], T=T, b1=REF["b1"], seed=0,
    )
    d = 201
    expected_q = max(1, math.floor(
        REF["n"] ** 2 * 0.1 ** 2
        / (REF["L1"] ** 2 * T * d * math.log(1.0 / REF_DELTA))
    ))
    assert cfg.q == expected_q


# ---------------------------------------------------------------------------
# q lower-guard to >= 1
# ---------------------------------------------------------------------------

def test_q_lower_guarded_to_one():
    # raw q = 0.012 -> guarded up to 1; all other guards pass
    cfg = resolve_config(
        _params_of_size(500), 5000, 1.0,
        epsilon=0.1, delta=1.0 / 5000,
        L0=1.0, L1=5.0, T=200, b1=5000, seed=0,
    )
    assert cfg.q == 1


# ---------------------------------------------------------------------------
# Strict in-regime guards each raise ValueError
# ---------------------------------------------------------------------------

def test_guard_q_gt_T():
    # reference eps=1.0 case: q=2941 > T=200
    with pytest.raises(ValueError, match="exceeds T="):
        resolve_config(
            _params_of_size(REF_D), REF["n"], REF["F0"],
            epsilon=1.0, delta=REF_DELTA,
            L0=REF["L0"], L1=REF["L1"], T=REF["T"], b1=REF["b1"], seed=0,
        )


def test_guard_b2_gt_n():
    # n=300, d=200, L0=5, L1=0.5, F0=0.01, eps=0.5 -> b2=459 > n=300,
    # while q<=T and n>=n_min.
    with pytest.raises(ValueError, match="exceeds n="):
        resolve_config(
            _params_of_size(200), 300, 0.01,
            epsilon=0.5, delta=1.0 / 300,
            L0=5.0, L1=0.5, T=500, b1=300, seed=0,
        )


def test_guard_n_lt_n_min():
    # tiny n far below n_min (~ sqrt(d)/eps with small eps)
    with pytest.raises(ValueError, match="below the Theorem B.3 in-regime"):
        resolve_config(
            _params_of_size(REF_D), 10, 1.0,
            epsilon=0.01, delta=1.0 / 10,
            L0=1.0, L1=5.0, T=200, b1=10, seed=0,
        )


# ---------------------------------------------------------------------------
# Missing non-derivable mandatory parameter errors as usual
# ---------------------------------------------------------------------------

def test_missing_mandatory_param_raises_type_error():
    with pytest.raises(TypeError):
        resolve_config(  # no epsilon
            _params_of_size(REF_D), REF["n"], REF["F0"],
            delta=REF_DELTA, L0=1.0, L1=5.0, T=200, b1=REF["n"], seed=0,
        )


# ---------------------------------------------------------------------------
# Provenance is printed (which params derived + which inputs used)
# ---------------------------------------------------------------------------

def test_provenance_printed_for_derived_params(capsys):
    resolve_config(
        _params_of_size(REF_D), REF["n"], REF["F0"],
        epsilon=0.1, delta=REF_DELTA,
        L0=REF["L0"], L1=REF["L1"], T=REF["T"], b1=REF["b1"], seed=0,
    )
    out = capsys.readouterr().out
    assert "derived" in out
    # values, not formulas
    assert "eta=" in out and "q=29" in out and "b2=" in out
    assert "n=800000" in out
