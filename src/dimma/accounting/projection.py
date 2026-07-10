"""Noise-scale calibration for the projection mechanism (Ghazi et al. 2024).

Algorithm 1 of Ghazi, Guzmán, Kamath, Kumar, Manurangsi, *"Differentially
Private Optimization with Sparse Gradients"* (NeurIPS 2024) privatises the
empirical mean of ``s``-sparse, ``l_2``-bounded (``<= L``) records by adding
coordinate-wise noise and projecting onto the ``l_1``-ball. The projection is
post-processing (privacy-free); **all** of the privacy comes from the noise
scales computed here.

Unlike ``dimma.core`` (which forbids DP claims), this module *is* the
accounting layer: each function is calibrated to a named sensitivity of the
empirical mean over neighbouring datasets (Appendix A, Fact A.1 of the paper)
and returns the exact scale that makes Algorithm 1 differentially private.

Sensitivities of ``z̄(S) = (1/n) Σ_i z_i`` over sparse neighbours ``S ≃ S'``:

- ``l_1``-sensitivity  ``Δ₁ = 2·L·√s / n``  (Laplace mechanism, pure-DP).
- ``l_2``-sensitivity  ``Δ₂ = 2·L / n``      (Gaussian mechanism, approx-DP).
"""

from __future__ import annotations

import math


def laplace_noise_scale(L: float, s: float, n: float, epsilon: float) -> float:
    """Laplace ``b`` parameter for ``ε``-DP mean release (pure-DP, ``δ = 0``).

    Returns the per-coordinate Laplace scale of Algorithm 1's pure-DP branch:

    .. math::

        b = \\frac{2 L \\sqrt{s}}{n \\, \\epsilon}

    This is the ``l_1``-sensitivity ``Δ₁ = 2 L √s / n`` of the empirical mean
    over sparse neighbouring datasets, divided by ``ε`` — the standard Laplace
    mechanism (Ghazi et al. 2024, Appendix A, Fact A.1). The returned value is
    the Laplace ``b`` parameter (density ``∝ exp(-|x| / b)``), **not** a
    standard deviation; feed it directly to
    :func:`dimma.core.add_pytree_laplace_noise`.

    Parameters
    ----------
    L : float
        Per-record ``l_2`` norm bound (the norm scale; often ``1``).
    s : float
        Sparsity bound — an upper bound on ``‖z_i‖_0``. Enters via ``√s`` from
        ``‖z‖_1 <= √s ‖z‖_2``.
    n : float
        Dataset size (number of records averaged).
    epsilon : float
        Target privacy budget ``ε``.

    Returns
    -------
    scale : float
        Laplace ``b`` parameter ``2 L √s / (n ε)``.
    """
    return 2.0 * L * math.sqrt(s) / (n * epsilon)


def gaussian_noise_scale(
    L: float, n: float, epsilon: float, delta: float
) -> float:
    """Gaussian ``σ`` (std) for ``(ε, δ)``-DP mean release (approx-DP, ``δ > 0``).

    Returns the per-coordinate Gaussian standard deviation of Algorithm 1's
    approximate-DP branch:

    .. math::

        \\sigma = \\frac{\\sqrt{8 \\ln(1.25/\\delta)} \\; L}{n \\, \\epsilon}
        \\qquad\\left( \\sigma^2 = \\frac{8 L^2 \\ln(1.25/\\delta)}{(n\\epsilon)^2} \\right)

    This is the classical Gaussian-mechanism calibration (Ghazi et al. 2024,
    Appendix A, Fact A.1) to the ``l_2``-sensitivity ``Δ₂ = 2 L / n`` of the
    empirical mean.

    .. note::

        **Valid only for ``ε ∈ (0, 1)``.** The classical Dwork–Roth calibration
        ``σ = √(2 ln(1.25/δ)) · Δ₂ / ε`` (equivalently the ``√(8 ln(1.25/δ))``
        form above with ``Δ₂ = 2 L / n``) certifies ``(ε, δ)``-DP only for
        ``ε ∈ (0, 1)``; for ``ε ≥ 1`` the returned ``σ`` under-noises and the
        release is **not** ``(ε, δ)``-DP. This function is a pure closed-form
        scale and does **not** validate ``ε`` (mirroring
        :func:`laplace_noise_scale`); the eager ``ε < 1`` guard lives in
        :func:`dimma.mechanisms.projection.projection_mechanism`, which is the
        layer that claims DP. If all-``ε`` support is ever needed, switch to the
        analytic Gaussian mechanism (Balle & Wang 2018), which calibrates
        exactly for every ``ε > 0``.

    .. note::

        **The Gaussian scale does NOT depend on the sparsity ``s``.** The
        Laplace branch uses the ``l_1``-sensitivity ``Δ₁ = 2 L √s / n``, which
        carries a ``√s``; the Gaussian branch uses the ``l_2``-sensitivity
        ``Δ₂ = 2 L / n``, which does not. This asymmetry routinely trips
        readers up — a single ``s``-sparse record can change ``z̄`` by at most
        ``2L/n`` in ``l_2`` regardless of ``s``, so ``s`` appears only in the
        projection radius ``L√s``, never in this scale.

    Parameters
    ----------
    L : float
        Per-record ``l_2`` norm bound (the norm scale; often ``1``).
    n : float
        Dataset size (number of records averaged).
    epsilon : float
        Target privacy budget ``ε``. The classical calibration is valid only
        for ``ε ∈ (0, 1)`` (see note); the caller
        (:func:`dimma.mechanisms.projection.projection_mechanism`) enforces this.
    delta : float
        Target privacy failure probability ``δ`` (must be ``> 0`` here).

    Returns
    -------
    scale : float
        Gaussian standard deviation ``√(8 ln(1.25/δ)) · L / (n ε)``.
    """
    return math.sqrt(8.0 * math.log(1.25 / delta)) * L / (n * epsilon)
