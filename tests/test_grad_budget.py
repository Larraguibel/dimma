"""Tests for expected_grad_evals in examples/private_spiderboost/lib/model.py.

The formula (fixed by ADR-0001):
    n_anchors   = T // q + 1
    n_variation = T - T // q
    return n_anchors * b1 + n_variation * 2 * b2
"""

import sys
from pathlib import Path

LIB_ROOT = Path(__file__).parent.parent / "examples" / "private_spiderboost" / "lib"
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))

from model import expected_grad_evals


# ---------------------------------------------------------------------------
# Representative values — hand-computed
# ---------------------------------------------------------------------------

def test_worked_example_from_spec():
    # T=10, q=3, b1=256, b2=16
    # n_anchors   = 10//3 + 1 = 3 + 1 = 4
    # n_variation = 10 - 3    = 7
    # result      = 4*256 + 7*2*16 = 1024 + 224 = 1248
    assert expected_grad_evals(T=10, q=3, b1=256, b2=16) == 1248


def test_exact_multiple_of_q():
    # T=12, q=4, b1=100, b2=10
    # n_anchors   = 12//4 + 1 = 3 + 1 = 4
    # n_variation = 12 - 3    = 9
    # result      = 4*100 + 9*2*10 = 400 + 180 = 580
    assert expected_grad_evals(T=12, q=4, b1=100, b2=10) == 580


def test_q_larger_than_T():
    # T=5, q=10, b1=50, b2=5
    # n_anchors   = 5//10 + 1 = 0 + 1 = 1
    # n_variation = 5 - 0     = 5
    # result      = 1*50 + 5*2*5 = 50 + 50 = 100
    assert expected_grad_evals(T=5, q=10, b1=50, b2=5) == 100


def test_q_equals_one():
    # T=6, q=1, b1=200, b2=20
    # n_anchors   = 6//1 + 1 = 6 + 1 = 7   (every step is an anchor)
    # n_variation = 6 - 6    = 0
    # result      = 7*200 + 0*2*20 = 1400
    assert expected_grad_evals(T=6, q=1, b1=200, b2=20) == 1400


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_T_equals_zero():
    # T=0, q=anything (say 3): only step t=0 fires, which is an anchor
    # n_anchors   = 0//3 + 1 = 0 + 1 = 1
    # n_variation = 0 - 0    = 0
    # result      = 1*b1
    assert expected_grad_evals(T=0, q=3, b1=128, b2=32) == 128
    assert expected_grad_evals(T=0, q=1, b1=64, b2=8)   == 64


def test_T_equals_q():
    # T=5, q=5, b1=300, b2=15
    # n_anchors   = 5//5 + 1 = 1 + 1 = 2
    # n_variation = 5 - 1    = 4
    # result      = 2*300 + 4*2*15 = 600 + 120 = 720
    assert expected_grad_evals(T=5, q=5, b1=300, b2=15) == 720


def test_T_not_multiple_of_q():
    # T=7, q=3, b1=256, b2=32
    # n_anchors   = 7//3 + 1 = 2 + 1 = 3
    # n_variation = 7 - 2    = 5
    # result      = 3*256 + 5*2*32 = 768 + 320 = 1088
    assert expected_grad_evals(T=7, q=3, b1=256, b2=32) == 1088


# ---------------------------------------------------------------------------
# Budget symmetry: pure function — same args, same result regardless of
# whether the caller represents a DP or non-DP run.
# ---------------------------------------------------------------------------

def test_dp_and_non_dp_yield_same_budget():
    """expected_grad_evals is a pure function of (T, q, b1, b2).

    A "DP config" and a "non-DP baseline config" with identical hyperparameters
    must produce the same gradient budget — the function is stateless.
    """
    T, q, b1, b2 = 20, 5, 512, 64

    dp_budget      = expected_grad_evals(T=T, q=q, b1=b1, b2=b2)
    baseline_budget = expected_grad_evals(T=T, q=q, b1=b1, b2=b2)

    assert dp_budget == baseline_budget


def test_return_type_is_int():
    result = expected_grad_evals(T=10, q=3, b1=256, b2=16)
    assert isinstance(result, int)
