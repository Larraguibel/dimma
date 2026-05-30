"""Vanilla-JAX MLP with LayerNorm.

Parameters are stored as a plain pytree (a list of per-layer dicts plus a
final-layer dict). All functions are pure and side-effect free, which makes
them trivially compatible with ``jax.vmap`` / ``jax.grad`` / ``jax.jit`` and
with the per-sample gradient pattern required by Algorithm 2 of
Arora et al. (ICML 2023).
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np


Params = dict  # Per-layer dict; full params is a list[Params] + final dict.


def init_params(
    key: jax.Array,
    input_dim: int,
    hidden_dims: Sequence[int],
) -> dict:
    """Initialise MLP parameters.

    Architecture: ``input -> [Dense -> LayerNorm -> sigmoid] * len(hidden_dims) -> Dense(1)``.
    Dense weights use He initialization; biases start at 0; LayerNorm scale at
    1 and shift at 0.

    Parameters
    ----------
    key : jax.Array
        PRNG key used for weight initialization.
    input_dim : int
        Number of input features.
    hidden_dims : Sequence[int]
        Width of each hidden layer; ``len(hidden_dims)`` is the number of
        hidden Dense layers. Example: ``[64, 32]``.

    Returns
    -------
    params : dict
        A dict with keys ``"hidden"`` (list of per-layer dicts, each with
        ``W``, ``b``, ``ln_scale``, ``ln_shift``) and ``"out"`` (dict with
        ``W`` of shape ``(last_hidden, 1)`` and ``b`` of shape ``(1,)``).

    Notes
    -----
    Weight shapes
        ``W : (in_dim, out_dim)``, ``b : (out_dim,)``,
        ``ln_scale, ln_shift : (out_dim,)``.
    """
    hidden_dims = list(hidden_dims)
    layers: list[dict] = []
    prev = input_dim
    for h in hidden_dims:
        key, sub = jax.random.split(key)
        W = jax.random.normal(sub, (prev, h)) * jnp.sqrt(2.0 / prev)
        layers.append({
            "W": W,
            "b": jnp.zeros((h,)),
            "ln_scale": jnp.ones((h,)),
            "ln_shift": jnp.zeros((h,)),
        })
        prev = h
    key, sub = jax.random.split(key)
    out = {
        "W": jax.random.normal(sub, (prev, 1)) * jnp.sqrt(2.0 / prev),
        "b": jnp.zeros((1,)),
    }
    return {"hidden": layers, "out": out}


def _layer_norm(x: jax.Array, scale: jax.Array, shift: jax.Array,
                eps: float = 1e-5) -> jax.Array:
    """Apply LayerNorm to the last axis of ``x``."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return x_hat * scale + shift


def forward(params: dict, x: jax.Array) -> jax.Array:
    """Compute MLP logits for a single example or a batch.

    Parameters
    ----------
    params : dict
        Parameter pytree as returned by :func:`init_params`.
    x : jax.Array
        Either shape ``(d,)`` for a single example or ``(B, d)`` for a batch.

    Returns
    -------
    logit : jax.Array
        Raw (pre-sigmoid) logit. Shape ``()`` for a single example or
        ``(B,)`` for a batch.
    """
    h = x
    for layer in params["hidden"]:
        h = h @ layer["W"] + layer["b"]
        h = _layer_norm(h, layer["ln_scale"], layer["ln_shift"])
        h = jax.nn.gelu(h)
    logit = (h @ params["out"]["W"] + params["out"]["b"]).squeeze(-1)
    return logit


def per_sample_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Sigmoid binary cross-entropy loss for a single example.

    The loss is computed in a numerically-stable form, equivalent to
    ``-y * log(sigmoid(logit)) - (1 - y) * log(1 - sigmoid(logit))`` but
    safe for large ``|logit|``.

    Parameters
    ----------
    params : dict
        Parameter pytree.
    x : jax.Array, shape (d,)
        A single feature vector.
    y : jax.Array, shape ()
        Binary label in {0., 1.}.

    Returns
    -------
    loss : jax.Array, shape ()
        Scalar BCE loss for this example. Suitable for
        ``jax.vmap(jax.grad(per_sample_bce_loss), in_axes=(None, 0, 0))``.
    """
    logit = forward(params, x)
    return jnp.maximum(logit, 0.0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit)))


def batch_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean BCE loss over a batch (for monitoring, not for DP gradients).

    Parameters
    ----------
    params : dict
        Parameter pytree.
    x : jax.Array, shape (B, d)
        Batch of feature vectors.
    y : jax.Array, shape (B,)
        Batch of binary labels.

    Returns
    -------
    loss : jax.Array, shape ()
        Mean BCE loss across the batch.
    """
    logits = forward(params, x)
    return jnp.mean(
        jnp.maximum(logits, 0.0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


# ---------------------------------------------------------------------------
# Non-private SPIDER training loop
# ---------------------------------------------------------------------------

class SpiderTrainHistory(NamedTuple):
    """Per-step record from train_spider.

    Attributes
    ----------
    grad_norm : list[float], length T+1
        Global L2 norm of the gradient estimate at each step.
    wall_time_s : list[float], length T+1
        Wall-clock seconds per step.
    output_step : int
        Step t* uniformly drawn from {1, ..., T} at which params_random
        was snapshotted (mirrors the DP random-output rule).
    """
    grad_norm: list
    wall_time_s: list
    output_step: int


class SpiderTrainResult(NamedTuple):
    """Output of train_spider.

    Attributes
    ----------
    params_final : pytree
        Parameters after step T.
    params_random : pytree
        Parameters at the uniformly random output step t* ∈ {1, ..., T}.
        Mirrors the output rule of Private SpiderBoost for fair comparison.
    history : SpiderTrainHistory
    """
    params_final: Any
    params_random: Any
    history: SpiderTrainHistory


def train_spider(
    x_train: jax.Array,
    y_train: jax.Array,
    init_params: Any,
    T: int,
    q: int,
    b1: int,
    b2: int,
    eta: float,
    seed: int,
) -> SpiderTrainResult:
    """Non-private SPIDER training loop with the same anchor/variation step
    structure as Private SpiderBoost, but without gradient clipping or noise.

    Anchor steps (t mod q == 0) compute a fresh mean gradient over a batch
    of size b1. Variation steps (t mod q != 0) accumulate the mean gradient
    difference over a batch of size b2, giving a variance-reduced estimate.
    Parameters are updated via SGD: w_{t+1} = w_t - eta * grad_estimate_t.

    The random output step t* is drawn from {1, ..., T} with the same RNG
    convention as dimma.train so that params_random is directly comparable
    to the DP run's params_random.

    Parameters
    ----------
    x_train, y_train : jax.Array
        Full training set.
    init_params : pytree
        Initial parameters (e.g. from model.init_params).
    T : int
        Number of iterations. The loop runs t = 0, ..., T.
    q : int
        Phase length. Anchor steps fire when t mod q == 0.
    b1, b2 : int
        Anchor and variation batch sizes (sampled without replacement).
    eta : float
        Learning rate.
    seed : int
        Master seed. Sampling uses JAX PRNG; output-step draw uses
        numpy.random.default_rng(seed + 7919) to match dimma.train.

    Returns
    -------
    SpiderTrainResult
    """
    sampling_key = jax.random.PRNGKey(seed)
    control_rng = np.random.default_rng(seed + 7919)
    n = int(x_train.shape[0])

    @jax.jit
    def anchor_grad(params, x_batch, y_batch):
        return jax.grad(batch_bce_loss)(params, x_batch, y_batch)

    @jax.jit
    def variation_delta(params, params_prev, x_batch, y_batch):
        g_t = jax.grad(batch_bce_loss)(params, x_batch, y_batch)
        g_prev = jax.grad(batch_bce_loss)(params_prev, x_batch, y_batch)
        return jax.tree.map(lambda a, b: a - b, g_t, g_prev)

    def _norm(tree):
        leaves = jax.tree.leaves(tree)
        return float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves)))

    output_step = int(control_rng.integers(low=1, high=T + 1))
    params = init_params
    params_prev = init_params
    grad_est: Any = None
    params_random = None

    grad_norm_hist: list = []
    wall_time_hist: list = []

    for t in range(T + 1):
        t_start = time.perf_counter()
        sampling_key, subkey = jax.random.split(sampling_key)
        is_anchor = (t % q == 0)

        if is_anchor:
            idx = np.asarray(
                jax.random.choice(subkey, n, shape=(min(b1, n),), replace=False)
            )
            x_b, y_b = x_train[idx], y_train[idx]
            grad_est = anchor_grad(params, x_b, y_b)
        else:
            idx = np.asarray(
                jax.random.choice(subkey, n, shape=(min(b2, n),), replace=False)
            )
            x_b, y_b = x_train[idx], y_train[idx]
            delta = variation_delta(params, params_prev, x_b, y_b)
            grad_est = jax.tree.map(lambda a, b: a + b, grad_est, delta)

        params_prev = params
        params = jax.tree.map(lambda p, g: p - eta * g, params, grad_est)

        if t == output_step:
            params_random = params

        grad_norm_hist.append(_norm(grad_est))
        wall_time_hist.append(time.perf_counter() - t_start)

    assert params_random is not None, (
        "Internal error: output_step was never reached. "
        "Check loop range and output_step draw."
    )

    return SpiderTrainResult(
        params_final=params,
        params_random=params_random,
        history=SpiderTrainHistory(
            grad_norm=grad_norm_hist,
            wall_time_s=wall_time_hist,
            output_step=output_step,
        ),
    )
