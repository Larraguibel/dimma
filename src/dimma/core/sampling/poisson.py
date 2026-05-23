"""Poisson subsampling primitives.

Two variants are provided so call sites are greppable:

- :func:`poisson_subsample` — strict; raises on oversize batches.
- :func:`poisson_subsample_truncated` — deterministically truncates on
  oversize batches. Modified mechanism; heuristic accountant.

Both use a ``numpy.random.Generator`` (not a JAX PRNGKey) because the
output cardinality is data-dependent and therefore not JIT-friendly.
"""

from __future__ import annotations

import math

import numpy as np


def poisson_padded_batch_size(b_expected: int, n: int,
                              margin_sigmas: float = 6.0) -> int:
    """Choose a padding cap b_max for a Poisson-subsampled batch.

    Sets b_max = b_expected + margin_sigmas * sqrt(b_expected * (1 - p))
    where p = b_expected / n. With margin_sigmas = 6 the probability of a
    draw exceeding the cap is below ~10^-9 for the parameter ranges used
    in typical DP-SGD configurations.

    Parameters
    ----------
    b_expected : int
        Expected (mean) Poisson batch size. Equals n * p.
    n : int
        Training set size.
    margin_sigmas : float, default 6.0
        Safety margin in standard deviations.

    Returns
    -------
    b_max : int
        Padded batch size to use as a static shape in JIT-compiled kernels.

    Notes
    -----
    This is a *padding* cap, not a privacy parameter. It determines how
    many slots the JIT-compiled batch tensor reserves. Setting it too low
    causes either truncation (with poisson_subsample_truncated) or an
    exception (with poisson_subsample); setting it too high wastes memory
    but is privacy-safe.
    """
    p = b_expected / n
    std = math.sqrt(b_expected * max(1.0 - p, 0.0))
    return int(math.ceil(b_expected + margin_sigmas * std + 4))


def poisson_subsample(rng: np.random.Generator, n: int, p: float,
                      b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Strict Poisson subsampling for DP-SGD-style mechanisms.

    This is the standard Poisson-subsampled mechanism: each training example
    is independently included in the batch with probability p. The padded
    output is JIT-compatible — real entries occupy the first k slots, the
    rest are zero-padded and masked out.

    Raises RuntimeError on oversize batches rather than truncating. This
    matches the assumption of the standard Poisson-subsampled Gaussian
    privacy accounting (e.g. dp_accounting.PoissonSampledDpEvent).

    Parameters
    ----------
    rng : numpy.random.Generator
        NumPy RNG. Advances internally; the same generator should be used
        for all sampling steps to maintain independence across calls.
    n : int
        Training set size.
    p : float
        Per-example inclusion probability, in (0, 1].
    b_max : int
        Padding cap. Use poisson_padded_batch_size to compute a safe value.

    Returns
    -------
    indices : np.ndarray, shape (b_max,), dtype int64
        Padded indices into the training set. Slots beyond k are 0 (which
        are masked out and should not be interpreted as real samples).
    mask : np.ndarray, shape (b_max,), dtype float32
        1.0 for real entries, 0.0 for padding.

    Raises
    ------
    RuntimeError
        If the Bernoulli draw produces more than b_max selected samples.
        Indicates b_max was set too low; increase margin_sigmas or switch
        to poisson_subsample_truncated.

    Notes
    -----
    Privacy: This sampler is compatible with the standard
    PoissonSampledDpEvent accountant. The matching accountant in
    dimma.accounting.sampling is poisson_gaussian_epsilon.
    """
    bern = rng.random(n) < p
    idx = np.flatnonzero(bern)
    k = idx.size
    if k > b_max:
        raise RuntimeError(
            f"Poisson subsample drew {k} samples but b_max={b_max} "
            f"(n={n}, p={p}). Increase margin_sigmas in "
            f"poisson_padded_batch_size, or use poisson_subsample_truncated "
            f"if you accept the heuristic accountant."
        )
    pad = b_max - k
    indices = np.concatenate([idx, np.zeros(pad, dtype=np.int64)])
    mask = np.concatenate(
        [np.ones(k, dtype=np.float32), np.zeros(pad, dtype=np.float32)]
    )
    return indices, mask


def poisson_subsample_truncated(rng: np.random.Generator, n: int, p: float,
                                b_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Truncated Poisson subsampling — modified mechanism, heuristic accountant.

    WARNING
    -------
    This is NOT the standard Poisson-subsampled mechanism. On the rare event
    that the Bernoulli draw exceeds b_max, this function deterministically
    subsamples down to b_max via rng.choice. This truncation changes the
    mechanism's privacy guarantee:

    - The standard PoissonSampledDpEvent bound is a LOWER BOUND on the true
      privacy cost of this mechanism.
    - The true privacy cost is strictly higher (adversary advantage from
      truncation).
    - The matching accountant poisson_gaussian_truncated_epsilon returns the
      standard bound, labeled as a lower bound. Do not use this for privacy
      claims in published work without a tighter analysis.

    This primitive exists for two reasons:
    1. To match the behavior of the source implementation
       (private_spider_boost_criteo/src/train.py) so historical runs remain
       reproducible.
    2. To support exploratory research into modified-mechanism accountants.

    If you do not specifically need truncated semantics, use poisson_subsample
    instead.

    Parameters, Returns, Notes
    --------------------------
    Same as poisson_subsample, except no exception is raised on oversize
    batches.
    """
    bern = rng.random(n) < p
    idx = np.flatnonzero(bern)
    k = idx.size
    if k > b_max:
        idx = rng.choice(idx, size=b_max, replace=False)
        k = b_max
    pad = b_max - k
    indices = np.concatenate([idx, np.zeros(pad, dtype=np.int64)])
    mask = np.concatenate(
        [np.ones(k, dtype=np.float32), np.zeros(pad, dtype=np.float32)]
    )
    return indices, mask
