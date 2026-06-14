"""Theorem B.3 config resolver for Private SpiderBoost.

``resolve_config`` fills the proof-prescribed hyperparameters ``{eta, q, b2}``
from Theorem B.3 of Arora et al. (2023), "Faster Rates of Convergence to
Stationary Points in Differentially Private Optimization" (p. 16), and returns a
fully-concrete :class:`~dimma.algorithms.spiderboost.train.TrainConfig`.

It is a pure pre-``train`` layer that mirrors the existing
"call ``compute_noise_scales`` before ``train``" idiom: ``TrainConfig``,
``compute_noise_scales`` and ``train`` are unchanged. See
``docs/adr/0002-thm-b3-config-resolver.md`` for the full design.

Derived quantities (verified against Theorem B.3, p. 16)::

    eta = 1 / (2·L1)
    q   = floor( n²·ε² / (L1²·T·d·log(1/δ)) )
    b2  = floor( max{ (L0·n·ε / sqrt(F0·L1·d·log(1/δ)))^(2/3),
                      (L0·n·d·log(1/δ))^(1/3) / ((L1·F0)^(1/6)·ε^(2/3)) } )

The ``q`` formula follows the theorem **statement**, which keeps the ``L1²`` in
the denominator. The proof (p. 17) instead sets ``q = n²ε²/(T·d·log(1/δ))``
without ``L1²`` (because ``η=1/(2·L1)`` cancels it); the two disagree by a factor
of ``L1²``. We follow the statement — it is the formal claim and matches issue
#7's validated numeric table. See ADR-0002 §3.

``T`` and ``b1`` stay caller-provided (off-theory budget choices); ``L0`` (clip
threshold) and ``L1`` (smoothness) are required, documented inputs — the library
provides no estimator (ADR-0002 §7).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import jax

from dimma.algorithms.spiderboost.train import TrainConfig


def resolve_config(
    init_params: Any,
    n: int,
    F0: float,
    *,
    epsilon: float,
    delta: float,
    L0: float,
    L1: float,
    T: int,
    b1: int,
    seed: int,
    q: Optional[int] = None,
    b2: Optional[int] = None,
    eta: Optional[float] = None,
    margin_sigmas: float = 6.0,
) -> TrainConfig:
    """Resolve the Theorem-B.3-prescribed SpiderBoost hyperparameters.

    Any of ``eta``, ``q``, ``b2`` left ``None`` is derived from Theorem B.3; an
    explicit value passes through unchanged. The non-derivable mandatory
    parameters are keyword-only without defaults, so omitting one raises
    ``TypeError`` as usual. The model dimension ``d`` is computed from
    ``init_params``; ``F0`` (initial suboptimality ``F(w0;S)``, ``F*=0``
    convention) is supplied by the caller.

    Parameters
    ----------
    init_params : pytree
        Initial parameters. ``d = sum(leaf.size for leaf in tree_leaves)``.
    n : int
        Dataset size.
    F0 : float
        Initial suboptimality ``F(w0;S)`` (``F*=0`` convention). Positive.
    epsilon, delta : float
        Target privacy budget.
    L0 : float
        Per-sample gradient clipping threshold (anchor sensitivity).
    L1 : float
        Smoothness constant (variation-step sensitivity).
    T : int
        Number of phases (loop horizon). Caller-provided, not derived.
    b1 : int
        Anchor batch size. Caller-provided, not derived.
    seed : int
        Master seed forwarded to ``TrainConfig``.
    q, b2, eta : optional
        Phase length, variation batch size, learning rate. ``None`` ⇒ derive
        from Theorem B.3; explicit ⇒ used as given.
    margin_sigmas : float, default 6.0
        Forwarded to ``TrainConfig`` (Poisson padding safety margin).

    Returns
    -------
    TrainConfig
        Fully-concrete config, ready for ``compute_noise_scales`` then ``train``.

    Raises
    ------
    ValueError
        If the resolved instance leaves Theorem B.3's stated regime:
        ``n < n_min``, ``q > T`` (anchor never refreshes within the horizon),
        or ``b2 > n`` (expected Poisson batch larger than the dataset). See
        ADR-0002 §5. ``q`` is lower-guarded to ``>= 1`` before the ``q > T``
        check.
    """
    d = int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(init_params)))
    log_inv_delta = math.log(1.0 / delta)

    # --- in-regime precondition: dataset must be large enough (ADR-0002 §5) ---
    n_min = max(
        (L0 * epsilon) ** 2 / (F0 * L1 * d * log_inv_delta),
        math.sqrt(d) * max(1.0, math.sqrt(L1 * F0) / L0) / epsilon,
    )
    if n < n_min:
        raise ValueError(
            f"n={n} is below the Theorem B.3 in-regime minimum n_min={n_min:.4g} "
            f"(computed from L0={L0}, L1={L1}, F0={F0}, d={d}, epsilon={epsilon}, "
            f"delta={delta}). Increase n, raise epsilon, or relax L0/L1/F0 to "
            f"enter the regime the proof covers."
        )

    derived: list[str] = []

    # --- eta = 1 / (2 L1) ---
    if eta is None:
        eta = 1.0 / (2.0 * L1)
        derived.append(f"eta={eta:.6g} from L1={L1}")

    # --- q = floor( n^2 eps^2 / (L1^2 T d log(1/delta)) ), lower-guarded to >=1 ---
    if q is None:
        q = math.floor(
            (n ** 2 * epsilon ** 2) / (L1 ** 2 * T * d * log_inv_delta)
        )
        q = max(1, q)
        derived.append(
            f"q={q} from n={n}, epsilon={epsilon}, L1={L1}, T={T}, d={d}, "
            f"delta={delta}"
        )

    # --- b2 = floor(max{ term_a, term_b }) ---
    if b2 is None:
        term_a = (
            L0 * n * epsilon / math.sqrt(F0 * L1 * d * log_inv_delta)
        ) ** (2.0 / 3.0)
        term_b = (
            (L0 * n * d * log_inv_delta) ** (1.0 / 3.0)
            / ((L1 * F0) ** (1.0 / 6.0) * epsilon ** (2.0 / 3.0))
        )
        b2 = math.floor(max(term_a, term_b))
        derived.append(
            f"b2={b2} from n={n}, epsilon={epsilon}, L0={L0}, L1={L1}, F0={F0}, "
            f"d={d}, delta={delta}"
        )

    # --- post-resolution in-regime guards (ADR-0002 §5) ---
    if q > T:
        raise ValueError(
            f"Resolved q={q} exceeds T={T}: the phase length is longer than the "
            f"horizon, so the anchor never refreshes. Raise T (Theorem B.3 needs "
            f"T >= q) or lower epsilon to bring q into range."
        )
    if b2 > n:
        raise ValueError(
            f"Resolved b2={b2} exceeds n={n}: the expected Poisson variation "
            f"batch is larger than the dataset (ill-defined). Lower epsilon or "
            f"adjust L0/L1/F0 to bring b2 into range."
        )

    # --- provenance: which params were derived, from which input values ---
    if derived:
        print("resolve_config: derived " + "; ".join(derived))
    else:
        print("resolve_config: no parameters derived (eta, q, b2 all provided)")

    return TrainConfig(
        epsilon=epsilon,
        delta=delta,
        L0=L0,
        L1=L1,
        T=T,
        q=q,
        b1=b1,
        b2=b2,
        eta=eta,
        seed=seed,
        margin_sigmas=margin_sigmas,
    )
