"""Plotting helpers for the Private SpiderBoost demo.

Each function builds one figure, saves it under ``figs/``, and returns the
``matplotlib.figure.Figure`` object so the notebook can also display it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def _ensure_parent(save_path: str | Path) -> Path:
    """Create the parent directory of ``save_path`` if missing; return Path."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_training_loss(loss_history: Sequence[float],
                       save_path: str | Path,
                       window: int = 50) -> plt.Figure:
    """Plot mean BCE loss vs. SpiderBoost step with a rolling-window overlay.

    Parameters
    ----------
    loss_history : Sequence[float], length ``T + 1``
        Per-step training loss (anchor-batch BCE; see :class:`train.TrainHistory`).
    save_path : str or pathlib.Path
        Where the PNG is written.
    window : int, default 50
        Width of the rolling-mean window. Set ``window <= 1`` to disable the
        overlay.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure (kept open so the notebook can display it).

    Notes
    -----
    The y-axis is linear; for very long runs consider switching to log. The
    smoothed line uses ``numpy.convolve(..., mode='valid')`` so its length is
    ``N - window + 1``; it is plotted centered on the window midpoint
    ``(window - 1) / 2`` so it visually aligns with the raw curve.
    """
    save_path = _ensure_parent(save_path)
    loss = np.asarray(loss_history, dtype=float)
    n = len(loss)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(n), loss, lw=1.0, color="grey", alpha=0.3,
            label="per-step")
    if window > 1 and n >= window:
        kernel = np.ones(window, dtype=float) / window
        smoothed = np.convolve(loss, kernel, mode="valid")
        x_smoothed = np.arange(len(smoothed)) + (window - 1) / 2.0
        ax.plot(x_smoothed, smoothed, lw=1.6, color="tab:blue",
                label=f"rolling mean (w={window})")
        ax.legend(loc="upper right")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean BCE on batch")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_test_loss(test_loss_steps: Sequence[int],
                   test_loss: Sequence[float],
                   save_path: str | Path) -> plt.Figure:
    """Plot mean BCE on the test set vs. step.

    Parameters
    ----------
    test_loss_steps : Sequence[int]
        SpiderBoost step at which each test loss was computed.
    test_loss : Sequence[float]
        Mean BCE values aligned with ``test_loss_steps``.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(test_loss_steps, test_loss, marker="o", lw=1.2, color="tab:red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean BCE on test set")
    ax.set_title("Test loss over training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_gradient_norm(grad_norm_history: Sequence[float],
                       save_path: str | Path) -> plt.Figure:
    """Plot the empirical running gradient norm ``||∇_t||`` (log y) vs. step.

    Parameters
    ----------
    grad_norm_history : Sequence[float], length ``T + 1``
        Running estimate ``||∇_t||`` returned by Algorithm 2 each step.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    This is the *noisy* SpiderBoost estimate — it includes the variation
    accumulator and the Gaussian noise — *not* the true population
    gradient. It is the natural quantity the algorithm tracks.
    """
    save_path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    arr = np.asarray(grad_norm_history)
    arr = np.where(arr > 0, arr, np.nan)  # safe log
    ax.plot(np.arange(len(arr)), arr, lw=1.0, color="tab:orange")
    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\|\nabla_t\|_2$ (log scale)")
    ax.set_title("Empirical gradient norm")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                   auc_value: float, save_path: str | Path) -> plt.Figure:
    """Plot the ROC curve on the test set.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_test,)
        Binary labels in {0, 1}.
    y_scores : np.ndarray, shape (n_test,)
        Continuous scores (logits or probabilities).
    auc_value : float
        ROC-AUC, displayed in the legend.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_scores))
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot(fpr, tpr, lw=1.6, color="tab:blue",
            label=f"AUC = {auc_value:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.8, ls="--", label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test ROC curve")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_auc_history(eval_steps: Sequence[int], eval_auc: Sequence[float],
                     save_path: str | Path) -> plt.Figure:
    """Plot test ROC-AUC vs. step (one point per anchor cycle).

    Parameters
    ----------
    eval_steps : Sequence[int]
        SpiderBoost step at which each AUC was computed.
    eval_auc : Sequence[float]
        ROC-AUC values aligned with ``eval_steps``.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eval_steps, eval_auc, marker="o", lw=1.2, color="tab:green")
    ax.set_xlabel("Step")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Test ROC-AUC over training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_epsilon_sweep(epsilons: Sequence[float],
                       auc_random: Sequence[float],
                       auc_final: Sequence[float],
                       save_path: str | Path,
                       auc_baseline: float | None = None) -> plt.Figure:
    """Plot test ROC-AUC vs. privacy budget ε (privacy-utility tradeoff).

    Parameters
    ----------
    epsilons : Sequence[float]
        The ε values that were swept.
    auc_random : Sequence[float]
        Test ROC-AUC of the random iterate ``w̄`` (Algorithm 2 output rule),
        aligned with ``epsilons``.
    auc_final : Sequence[float]
        Test ROC-AUC of the final iterate ``w_T``, aligned with ``epsilons``.
    save_path : str or pathlib.Path
        Output PNG location.
    auc_baseline : float or None, default None
        Test ROC-AUC of the non-private SPIDER baseline (horizontal reference
        line). When provided, draws a dashed red line labeled
        "non-private baseline". When ``None``, the plot is unchanged.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epsilons, auc_random, marker="o", lw=1.4, color="tab:green",
            label=r"$\bar w$ (random iterate)")
    ax.plot(epsilons, auc_final, marker="s", lw=1.4, color="tab:blue",
            label=r"$w_T$ (final iterate)")
    if auc_baseline is not None:
        ax.axhline(auc_baseline, ls="--", lw=1.2, color="tab:red",
                   label="non-private baseline")
    ax.set_xscale("log")
    ax.set_xlabel(r"Privacy budget $\varepsilon$ (log scale)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Privacy-utility tradeoff")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_q_sweep(q_values: Sequence[int],
                 auc_random: Sequence[float],
                 auc_final: Sequence[float],
                 save_path: str | Path) -> plt.Figure:
    """Plot test ROC-AUC vs. phase length ``q``.

    Parameters
    ----------
    q_values : Sequence[int]
        The phase-length values that were swept.
    auc_random : Sequence[float]
        Test ROC-AUC of ``w̄`` aligned with ``q_values``.
    auc_final : Sequence[float]
        Test ROC-AUC of ``w_T`` aligned with ``q_values``.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(q_values, auc_random, marker="o", lw=1.4, color="tab:green",
            label=r"$\bar w$ (random iterate)")
    ax.plot(q_values, auc_final, marker="s", lw=1.4, color="tab:blue",
            label=r"$w_T$ (final iterate)")
    ax.set_xlabel("Phase length $q$")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Phase-length tradeoff (fixed $\\varepsilon$)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_delta_sweep(
    exponents: Sequence[float],
    auc_random: Sequence[float],
    auc_final: Sequence[float],
    auc_baseline: float,
    save_path: str | Path,
) -> plt.Figure:
    """Plot test ROC-AUC vs. δ exponent α (where δ = 1/n^α).

    Parameters
    ----------
    exponents : Sequence[float]
        The α values that were swept.
    auc_random : Sequence[float]
        Test ROC-AUC of the random iterate w̄ per α.
    auc_final : Sequence[float]
        Test ROC-AUC of the final iterate w_T per α.
    auc_baseline : float
        Test ROC-AUC of the non-private JAX baseline (horizontal reference line).
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(exponents, auc_random, marker="o", lw=1.4, color="tab:green",
            label=r"$\bar w$ (random iterate)")
    ax.plot(exponents, auc_final, marker="s", lw=1.4, color="tab:blue",
            label=r"$w_T$ (final iterate)")
    ax.axhline(auc_baseline, ls="--", lw=1.2, color="tab:red",
               label="non-private baseline")
    ax.set_xlabel(r"$\delta$ exponent $\alpha$  ($\delta = 1/n^\alpha$)")
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title(r"Effect of $\delta$ on utility  ($\varepsilon = 1.0$ fixed)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_grad_norm_comparison(
    dp_grad_norm: Sequence[float],
    dp_cumulative_evals: Sequence[int],
    baseline_grad_norm: Sequence[float],
    baseline_cumulative_evals: Sequence[int],
    save_path: str | Path,
) -> plt.Figure:
    """Plot DP and non-private SPIDER gradient-norm trajectories on a shared log-y axis.

    The x-axis is cumulative expected gradient evaluations (computed per step via
    ``expected_grad_evals``), making anchor steps and variation steps commensurable
    despite their very different per-step costs.  The two curves are visually
    distinguishable by color and line style.

    Parameters
    ----------
    dp_grad_norm : Sequence[float], length T+1
        Per-step gradient-norm history from the DP Private SpiderBoost run.
    dp_cumulative_evals : Sequence[int], length T+1
        Cumulative expected gradient evaluations at each step of the DP run.
    baseline_grad_norm : Sequence[float], length T+1
        Per-step gradient-norm history from the non-private SPIDER baseline run.
    baseline_cumulative_evals : Sequence[int], length T+1
        Cumulative expected gradient evaluations at each step of the baseline run.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Both histories are expected to have the same length (same T, q, b1, b2) so the
    cumulative x-axis is identical; they are plotted on the same axes for direct
    comparison.  Values <= 0 are masked to NaN for safe log scaling.
    """
    save_path = _ensure_parent(save_path)
    dp_arr = np.asarray(dp_grad_norm, dtype=float)
    dp_arr = np.where(dp_arr > 0, dp_arr, np.nan)
    bl_arr = np.asarray(baseline_grad_norm, dtype=float)
    bl_arr = np.where(bl_arr > 0, bl_arr, np.nan)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(
        np.asarray(dp_cumulative_evals), dp_arr,
        lw=1.0, color="tab:orange", ls="-",
        label="Private SpiderBoost (DP)",
    )
    ax.plot(
        np.asarray(baseline_cumulative_evals), bl_arr,
        lw=1.0, color="tab:blue", ls="--",
        label="Non-private SPIDER baseline",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative expected gradient evaluations")
    ax.set_ylabel(r"$\|\nabla_t\|_2$ (log scale)")
    ax.set_title("Gradient norm: DP vs. non-private SPIDER")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_error_vs_dimension(
    dimensions: Sequence[int],
    projected_mean: Sequence[float],
    projected_std: Sequence[float],
    unprojected_mean: Sequence[float],
    unprojected_std: Sequence[float],
    save_path: str | Path,
) -> plt.Figure:
    """Plot ``l_2`` estimation error vs. dimension ``d`` on log-log axes.

    This is the **dimension-independence** figure for the projection mechanism
    (Ghazi et al. 2024, Algorithm 1). Two series are drawn against the ambient
    dimension ``d``, each with a mean ± std band across trials:

    - **projected** — error of the perturb-then-project estimate ``ẑ``. Nearly
      flat in ``d`` (grows only poly-logarithmically), because Lemma 3.1 bounds
      it by ``‖ξ‖_∞``, not ``‖ξ‖_2``.
    - **unprojected** — error of the noisy mean ``z̃`` with no projection. Grows
      like ``√d`` (the full noise magnitude), a straight line of slope ``1/2``
      on log-log axes.

    Parameters
    ----------
    dimensions : Sequence[int]
        The ambient dimensions ``d`` that were swept (x-axis).
    projected_mean, projected_std : Sequence[float]
        Mean and standard deviation (across trials) of ``‖ẑ − z̄‖_2`` per ``d``.
    unprojected_mean, unprojected_std : Sequence[float]
        Mean and standard deviation of ``‖z̃ − z̄‖_2`` per ``d``.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    The ± std bands are clipped at a small positive floor before plotting so the
    log-scaled ``fill_between`` does not receive non-positive values.
    """
    save_path = _ensure_parent(save_path)
    d = np.asarray(dimensions, dtype=float)
    pm = np.asarray(projected_mean, dtype=float)
    ps = np.asarray(projected_std, dtype=float)
    um = np.asarray(unprojected_mean, dtype=float)
    us = np.asarray(unprojected_std, dtype=float)
    floor = 1e-12

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(d, um, marker="s", lw=1.6, color="tab:red",
            label="unprojected (noisy mean)")
    ax.fill_between(d, np.maximum(um - us, floor), um + us,
                    color="tab:red", alpha=0.2)
    ax.plot(d, pm, marker="o", lw=1.6, color="tab:blue",
            label="projected (mechanism)")
    ax.fill_between(d, np.maximum(pm - ps, floor), pm + ps,
                    color="tab:blue", alpha=0.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Dimension $d$ (log scale)")
    ax.set_ylabel(r"$\|\hat z - \bar z\|_2$ (log scale)")
    ax.set_title("Estimation error vs. dimension (dimension-independence)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_lemma31_bound(
    bound: Sequence[float],
    error: Sequence[float],
    save_path: str | Path,
    labels: Sequence[str] | None = None,
) -> plt.Figure:
    """Scatter realized ``l_2`` error against the Lemma 3.1 bound with a ``y=x`` line.

    Lemma 3.1 (Ghazi et al. 2024) gives the *deterministic* per-trial guarantee
    ``‖ẑ − z̄‖_2 <= √(2 L ‖ξ‖_∞ √s)``. This plots the realized error (y) against
    that bound (x); every point must sit **on or below** the ``y = x`` reference
    line for the lemma to hold.

    Parameters
    ----------
    bound : Sequence[float]
        The Lemma 3.1 bound ``√(2 L ‖ξ‖_∞ √s)`` per trial (x-axis).
    error : Sequence[float]
        The realized error ``‖ẑ − z̄‖_2`` per trial (y-axis).
    save_path : str or pathlib.Path
        Output PNG location.
    labels : Sequence[str] or None, default None
        Optional per-point group label (e.g. ``"Laplace"`` / ``"Gaussian"``).
        When given, points are colored and legended by group. When ``None`` all
        points share a single color.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    x = np.asarray(bound, dtype=float)
    y = np.asarray(error, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    if labels is None:
        ax.scatter(x, y, s=18, alpha=0.6, color="tab:blue")
    else:
        labels = np.asarray(labels)
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
        for i, g in enumerate(dict.fromkeys(labels.tolist())):
            m = labels == g
            ax.scatter(x[m], y[m], s=18, alpha=0.6,
                       color=palette[i % len(palette)], label=str(g))
        ax.legend(loc="upper left")
    hi = float(max(x.max(), y.max())) * 1.05 if x.size else 1.0
    ax.plot([0, hi], [0, hi], color="gray", lw=1.0, ls="--", label="_y=x")
    ax.text(0.98, 0.02, r"$y = x$ (Lemma 3.1)", transform=ax.transAxes,
            ha="right", va="bottom", color="gray")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Lemma 3.1 bound  $\sqrt{2 L \|\xi\|_\infty \sqrt{s}}$")
    ax.set_ylabel(r"Realized error  $\|\hat z - \bar z\|_2$")
    ax.set_title("Lemma 3.1: realized error vs. bound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_grad_sparsity_histogram(
    nnz_per_sample: Sequence[int],
    s: int,
    save_path: str | Path,
) -> plt.Figure:
    """Histogram of per-sample gradient nonzero counts, with a line at ``s``.

    Empirically confirms that the hashed-logreg per-sample gradient is sparse:
    each sample's global gradient has at most ``num_fields + num_dense + 1``
    nonzeros regardless of the (large) table size. A vertical line marks the
    sparsity bound ``s`` used as the projection radius input.

    Parameters
    ----------
    nnz_per_sample : Sequence[int]
        Number of nonzero entries in each per-sample flattened gradient.
    s : int
        The sparsity bound ``s`` (drawn as a vertical reference line).
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    save_path = _ensure_parent(save_path)
    nnz = np.asarray(nnz_per_sample, dtype=int)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    lo = int(nnz.min())
    hi = int(max(nnz.max(), s))
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    ax.hist(nnz, bins=bins, color="tab:blue", alpha=0.75,
            edgecolor="white", label="per-sample nonzeros")
    ax.axvline(s, color="tab:red", lw=1.6, ls="--",
               label=f"sparsity bound s = {s}")
    ax.set_xlabel("Nonzeros in per-sample gradient")
    ax.set_ylabel("Count")
    ax.set_title("Per-sample gradient sparsity")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return fig


def plot_hyperparameter_summary(config: dict,
                                save_path: str | Path) -> plt.Figure:
    """Render a run's hyperparameters as a one-page table figure.

    Parameters
    ----------
    config : dict
        Dictionary of ``{name: value}`` pairs. Values are stringified.
    save_path : str or pathlib.Path
        Output PNG location.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Notes
    -----
    Useful as a header card next to the loss/AUC/ROC figures so the
    figure folder is self-describing.
    """
    save_path = _ensure_parent(save_path)
    rows = [(str(k), str(v)) for k, v in config.items()]
    fig, ax = plt.subplots(figsize=(7, 0.4 + 0.32 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["hyperparameter", "value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    for j in (0, 1):
        cell = table[(0, j)]
        cell.set_facecolor("#dddddd")
        cell.set_text_props(weight="bold")
    ax.set_title("Run configuration", pad=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
