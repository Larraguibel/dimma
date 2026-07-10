"""The projection mechanism for DP mean estimation (Ghazi et al. 2024, Alg. 1).

``projection_mechanism`` privatises an **already-computed** empirical mean of
``s``-sparse, ``l_2``-bounded (``<= L``) records: it adds coordinate-wise noise
(Laplace for pure-DP ``δ = 0``, Gaussian for approximate-DP ``δ > 0``) and then
Euclidean-projects the noisy answer onto the ``l_1``-ball of radius ``L√s``.

The projection is post-processing of a private quantity, so it consumes no
privacy budget; the DP guarantee comes entirely from the noise, whose scale is
calibrated in :mod:`dimma.accounting.projection`. The projection is what buys
the nearly dimension-independent error of Lemma 3.1
(``‖ẑ − z̄‖_2 <= √(2 L ‖ξ‖_∞ √s)``).

Layering rationale (see :mod:`dimma.mechanisms`): :mod:`dimma.core` forbids DP
claims and :mod:`dimma.algorithms` is for training loops, so this one-shot
DP-claiming primitive lives in :mod:`dimma.mechanisms` — the seam a future
DP-SGD would call once per step.

Reference: Ghazi, Guzmán, Kamath, Kumar, Manurangsi, *"Differentially Private
Optimization with Sparse Gradients"*, NeurIPS 2024, Section 3.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from dimma.accounting.projection import (
    gaussian_noise_scale,
    laplace_noise_scale,
)
from dimma.core.noise import (
    add_pytree_gaussian_noise,
    add_pytree_laplace_noise,
)
from dimma.core.projection import project_l1_ball_pytree


class ProjectionOutput(NamedTuple):
    """Return value of :func:`projection_mechanism`.

    Attributes
    ----------
    zhat : pytree
        The private, feasible estimate ``ẑ`` (satisfies ``‖ẑ‖_1 <= L√s``);
        this is the released quantity. Same structure as the input ``mean``.
    z_tilde : pytree
        The pre-projection noisy vector ``z̃`` (same structure as ``mean``),
        from which the noise ``ξ = z̃ − mean`` can be recovered (needed by the
        Lemma 3.1 tests and the demo notebook) without re-deriving the RNG
        internals.
    """

    zhat: Any
    z_tilde: Any


def projection_mechanism(
    mean,
    *,
    epsilon: float,
    delta: float,
    n: float,
    L: float,
    s: float,
    key,
) -> ProjectionOutput:
    """Privatise an empirical mean via perturb-then-project (Algorithm 1).

    Adds calibrated coordinate-wise noise to ``mean`` and projects the result
    onto the ``l_1``-ball of radius ``L√s``. The branch is chosen at the Python
    level from the static float ``delta``:

    - ``delta == 0.0`` → **Laplace** branch (pure ``ε``-DP). Scale from
      :func:`dimma.accounting.projection.laplace_noise_scale`
      (``= 2 L √s / (n ε)``), added by
      :func:`dimma.core.add_pytree_laplace_noise`.
    - ``delta > 0.0``  → **Gaussian** branch (``(ε, δ)``-DP). Std from
      :func:`dimma.accounting.projection.gaussian_noise_scale`
      (``= √(8 ln(1.25/δ)) L / (n ε)``, note: **``s``-independent**), added by
      :func:`dimma.core.add_pytree_gaussian_noise`.

    Privacy comes only from the noise; the projection is post-processing.

    .. note::

        **The Gaussian branch requires ``ε < 1``.**
        :func:`dimma.accounting.projection.gaussian_noise_scale` returns the
        classical Dwork–Roth calibration ``σ = √(2 ln(1.25/δ)) · Δ₂ / ε`` (with
        ``l_2``-sensitivity ``Δ₂ = 2 L / n``). This is faithful to the paper's
        stated formula (Ghazi et al. 2024, Appendix A, Fact A.1), but the
        classical bound only certifies ``(ε, δ)``-DP for ``ε ∈ (0, 1)``. For
        ``ε ≥ 1`` the calibration under-noises, so the release would **not**
        satisfy ``(ε, δ)``-DP — a silent privacy violation. This branch
        therefore rejects ``ε ≥ 1`` eagerly rather than emit an under-noised
        vector. If all-``ε`` Gaussian support is ever needed, adopt the analytic
        Gaussian mechanism (Balle & Wang 2018), which calibrates exactly for
        every ``ε > 0``. The Laplace branch (``δ = 0``) has no such restriction
        and accepts any ``ε > 0``.

    Parameters
    ----------
    mean : jax.Array or pytree of jax.Array
        The **already-computed** empirical mean ``z̄(S)`` to privatise. A bare
        array is treated as a single-leaf pytree, so flat arrays and nested
        parameter pytrees share one code path. This function does **not**
        compute the mean — the caller passes it in.
    epsilon : float
        Target privacy budget ``ε``. Must be ``> 0``. In the **Gaussian branch**
        (``δ > 0``) it must additionally be ``< 1`` — the classical Dwork–Roth
        Gaussian calibration is only valid for ``ε ∈ (0, 1)`` (see the note
        above). The Laplace branch (``δ = 0``) accepts any ``ε > 0``.
    delta : float
        Target failure probability ``δ``. Must be in ``[0, 1)`` (``δ >= 1`` is
        meaningless for DP). ``0.0`` selects the pure-DP Laplace branch; any
        value in ``(0, 1)`` selects the Gaussian branch.
    n : float
        Dataset size the mean was averaged over. Must be ``>= 1``.
    L : float
        Per-record ``l_2`` norm bound (norm scale). Must be ``> 0``.
    s : float
        Sparsity bound (upper bound on ``‖z_i‖_0``). Sets the projection radius
        ``L√s`` and, in the Laplace branch, the noise scale. Must be ``>= 1``.
    key : jax.Array
        PRNG key for the noise draw.

    Returns
    -------
    ProjectionOutput
        A ``(zhat, z_tilde)`` named tuple: the private, feasible estimate ``ẑ``
        (satisfies ``‖ẑ‖_1 <= L√s``) and the pre-projection noisy vector ``z̃``
        (both the same structure as ``mean``). The noise is recoverable as
        ``ξ = z_tilde − mean``.

    Raises
    ------
    ValueError
        If ``epsilon <= 0``, ``epsilon >= 1`` in the Gaussian branch
        (``delta > 0``), ``delta`` is outside ``[0, 1)``, ``n < 1``, ``s < 1``
        or ``L <= 0``.
    """
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}.")
    if delta < 0.0 or delta >= 1.0:
        raise ValueError(f"delta must be in [0, 1), got {delta}.")
    if delta > 0.0 and epsilon >= 1.0:
        raise ValueError(
            "epsilon must be < 1 in the Gaussian branch (delta > 0), got "
            f"{epsilon}."
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if s < 1:
        raise ValueError(f"s must be >= 1, got {s}.")
    if L <= 0.0:
        raise ValueError(f"L must be > 0, got {L}.")

    if delta == 0.0:
        scale = laplace_noise_scale(L, s, n, epsilon)
        z_tilde = add_pytree_laplace_noise(mean, key, scale)
    else:
        std = gaussian_noise_scale(L, n, epsilon, delta)
        z_tilde = add_pytree_gaussian_noise(mean, key, std)

    radius = L * math.sqrt(s)
    zhat = project_l1_ball_pytree(z_tilde, radius)

    return ProjectionOutput(zhat, z_tilde)
