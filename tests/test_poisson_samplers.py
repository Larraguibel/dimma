import math
import sys
from pathlib import Path

import numpy as np
import pytest

from dimma.core.sampling import (
    poisson_padded_batch_size,
    poisson_subsample,
    poisson_subsample_truncated,
)

SOURCE_ROOT = Path(__file__).parent.parent.parent / "private_spider_boost_criteo"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def test_padded_batch_size_formula():
    b, n, m = 1000, 100000, 6.0
    p = b / n
    std = math.sqrt(b * (1.0 - p))
    expected = int(math.ceil(b + m * std + 4))
    assert poisson_padded_batch_size(b, n, m) == expected


def test_padded_batch_size_p_equals_one():
    b = 100
    assert poisson_padded_batch_size(b, b) >= b + 4


def test_poisson_subsample_mask_sum():
    rng = np.random.default_rng(0)
    n, p = 1000, 0.05
    b_max = poisson_padded_batch_size(int(n * p), n)
    indices, mask = poisson_subsample(rng, n, p, b_max)
    # Re-run with same seed to verify mask.sum() equals the Bernoulli draw count.
    rng2 = np.random.default_rng(0)
    expected_k = int((rng2.random(n) < p).sum())
    assert int(mask.sum()) == expected_k


def test_poisson_subsample_raises_on_oversize():
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError) as exc:
        poisson_subsample(rng, n=1000, p=0.05, b_max=5)
    msg = str(exc.value)
    assert "poisson_subsample_truncated" in msg
    assert "margin_sigmas" in msg


def test_truncated_caps_at_b_max():
    rng = np.random.default_rng(0)
    indices, mask = poisson_subsample_truncated(rng, n=1000, p=0.05, b_max=5)
    assert int(mask.sum()) == 5
    assert indices.shape == (5,)


def test_both_agree_below_cap():
    n, p = 1000, 0.05
    b_max = poisson_padded_batch_size(int(n * p), n)
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    i_a, m_a = poisson_subsample(rng_a, n, p, b_max)
    i_b, m_b = poisson_subsample_truncated(rng_b, n, p, b_max)
    assert np.array_equal(i_a, i_b)
    assert np.array_equal(m_a, m_b)


def test_output_dtype_and_shape():
    n, p = 500, 0.05
    b_max = poisson_padded_batch_size(int(n * p), n)
    for fn in (poisson_subsample, poisson_subsample_truncated):
        rng = np.random.default_rng(7)
        indices, mask = fn(rng, n, p, b_max)
        assert indices.dtype == np.int64
        assert mask.dtype == np.float32
        assert indices.shape == (b_max,)
        assert mask.shape == (b_max,)


def test_rng_advances():
    rng = np.random.default_rng(42)
    n, p = 500, 0.05
    b_max = poisson_padded_batch_size(int(n * p), n)
    i1, m1 = poisson_subsample(rng, n, p, b_max)
    i2, m2 = poisson_subsample(rng, n, p, b_max)
    assert not (np.array_equal(i1, i2) and np.array_equal(m1, m2))


def test_regression_against_source_no_truncation():
    from src.train import _sample_poisson_padded
    n, p = 1000, 0.05
    b_max = poisson_padded_batch_size(int(n * p), n)

    rng_a = np.random.default_rng(2024)
    rng_b = np.random.default_rng(2024)
    i_a, m_a = poisson_subsample(rng_a, n, p, b_max)
    i_b, m_b = _sample_poisson_padded(rng_b, n, p, b_max)
    assert np.array_equal(i_a, i_b)
    assert np.array_equal(m_a, m_b)


def test_regression_against_source_with_truncation():
    from src.train import _sample_poisson_padded
    n, p, b_max = 1000, 0.05, 5

    rng_src = np.random.default_rng(2024)
    i_src, m_src = _sample_poisson_padded(rng_src, n, p, b_max)

    rng_trunc = np.random.default_rng(2024)
    i_trunc, m_trunc = poisson_subsample_truncated(rng_trunc, n, p, b_max)
    assert np.array_equal(i_src, i_trunc)
    assert np.array_equal(m_src, m_trunc)

    rng_strict = np.random.default_rng(2024)
    with pytest.raises(RuntimeError):
        poisson_subsample(rng_strict, n, p, b_max)
