"""Vanilla-JAX MLP with LayerNorm.

Parameters are stored as a plain pytree (a list of per-layer dicts plus a
final-layer dict). All functions are pure and side-effect free, which makes
them trivially compatible with ``jax.vmap`` / ``jax.grad`` / ``jax.jit`` and
with the per-sample gradient pattern required by Algorithm 2 of
Arora et al. (ICML 2023).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


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
