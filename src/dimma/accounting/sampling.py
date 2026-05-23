"""Generic Poisson-subsampled Gaussian accountants.

One function per Poisson sampler in ``dimma.core.sampling``. The strict
sampler's accountant is the standard RDP bound; the truncated sampler's
accountant returns the same numerical value as a *lower bound* on the
true privacy cost (the mechanism is non-standard).
"""

from __future__ import annotations


def poisson_gaussian_epsilon(
    sampling_probability: float,
    noise_multiplier: float,
    num_compositions: int,
    target_delta: float,
) -> float | None:
    """Standard Poisson-subsampled Gaussian RDP epsilon.

    Matches the strict Poisson subsampler (dimma.core.sampling.poisson_subsample).
    Returns epsilon for ``num_compositions`` independent applications of the
    Poisson-subsampled Gaussian mechanism at the given sampling probability,
    noise multiplier (z = std / sensitivity), and target delta.

    Returns
    -------
    epsilon : float or None
        ``None`` if dp_accounting is not installed; otherwise the RDP epsilon
        at target_delta.

    Notes
    -----
    The noise multiplier z is defined as std / sensitivity, where std is the
    standard deviation of the Gaussian noise added to the *sum* (not the
    average) and sensitivity is the per-sample contribution bound (often
    called L0 in the SpiderBoost literature, C in the DP-SGD literature).
    """
    try:
        from dp_accounting import dp_event, rdp
    except ImportError:
        return None

    accountant = rdp.RdpAccountant()
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_probability=sampling_probability,
            event=dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier),
        ),
        count=num_compositions,
    )
    return float(accountant.get_epsilon(target_delta=target_delta))


def poisson_gaussian_truncated_epsilon(
    sampling_probability: float,
    noise_multiplier: float,
    num_compositions: int,
    target_delta: float,
) -> float | None:
    """HEURISTIC accountant for the truncated Poisson sampler. READ CAREFULLY.

    This function returns the *standard* Poisson-subsampled Gaussian RDP
    bound. For the truncated sampler
    (dimma.core.sampling.poisson_subsample_truncated), this bound is a LOWER
    BOUND on the true privacy cost.

    The true epsilon of the truncated mechanism is strictly larger than what
    this function returns. Tightening this bound is an open research
    question.

    DO NOT use the return value of this function as a privacy claim in
    published work without further analysis. Use it only:
    - as a sanity-check lower bound during development,
    - to compare configurations qualitatively (the bound's ordering is
      preserved under typical parameter changes, even if its absolute value
      is loose), or
    - as a starting point for a tighter custom analysis.

    Returns
    -------
    epsilon : float or None
        ``None`` if dp_accounting is not installed; otherwise a LOWER BOUND
        on the true RDP epsilon at target_delta.
    """
    try:
        from dp_accounting import dp_event, rdp
    except ImportError:
        return None

    accountant = rdp.RdpAccountant()
    accountant.compose(
        dp_event.PoissonSampledDpEvent(
            sampling_probability=sampling_probability,
            event=dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier),
        ),
        count=num_compositions,
    )
    return float(accountant.get_epsilon(target_delta=target_delta))
