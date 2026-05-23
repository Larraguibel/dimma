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
                       save_path: str | Path) -> plt.Figure:
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
