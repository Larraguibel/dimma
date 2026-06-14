"""Flax MLP with LayerNorm — a reference model shipped with dimma.

This is a :term:`reference model` (see ``CONTEXT.md``): a concrete network the
library ships so researchers have a testing model in hand. It is **not** part of
the architecture-agnostic algorithm — ``dimma.train`` never depends on it. Model
code lives here under ``dimma.models``; it must never appear under
``dimma.algorithms``.

Parameters are stored as a plain Flax param pytree (the ``'params'`` leaf of
the variable collection), which is fully compatible with ``jax.vmap``,
``jax.grad``, and ``jax.jit``, and with the per-sample gradient pattern
required by Algorithm 2 of Arora et al. (ICML 2023).
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Flax module
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """MLP with LayerNorm after each hidden layer.

    Architecture: ``input -> [Dense -> LayerNorm -> GELU] * len(hidden_dims) -> Dense(1)``.
    """
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass for a single example ``(d,)`` or batch ``(B, d)``.

        Returns a scalar logit for a single example or ``(B,)`` for a batch.
        """
        h = x
        for width in self.hidden_dims:
            h = nn.Dense(width, kernel_init=nn.initializers.he_normal())(h)
            h = nn.LayerNorm()(h)
            h = nn.gelu(h)
        logit = nn.Dense(1, kernel_init=nn.initializers.he_normal())(h)
        return logit.squeeze(-1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _mlp_from_params(params: dict) -> MLP:
    """Reconstruct an MLP module from its param pytree.

    Flax names layers ``Dense_0, Dense_1, ..., Dense_N``.  The last Dense is
    the scalar output layer (output dim 1); all preceding ones are hidden.
    ``hidden_dims`` is read from the kernel output dimensions.
    """
    n_dense = sum(1 for k in params if k.startswith('Dense_'))
    hidden_dims = tuple(
        params[f'Dense_{i}']['kernel'].shape[1]
        for i in range(n_dense - 1)
    )
    return MLP(hidden_dims=hidden_dims)


def init_params(
    key: jax.Array,
    input_dim: int,
    hidden_dims: Sequence[int],
) -> dict:
    """Initialise MLP parameters via Flax.

    Returns the ``'params'`` pytree (a nested dict of arrays), which is a
    plain JAX pytree compatible with ``jax.grad`` / ``jax.vmap`` / ``jax.jit``.
    """
    mlp = MLP(hidden_dims=tuple(hidden_dims))
    dummy = jnp.zeros((input_dim,))
    variables = mlp.init(key, dummy)
    return variables['params']


def forward(params: dict, x: jax.Array) -> jax.Array:
    """Compute MLP logits for a single example ``(d,)`` or batch ``(B, d)``.

    The architecture is inferred directly from ``params`` — no global state.

    Parameters
    ----------
    params : dict
        Flax param pytree as returned by :func:`init_params`.
    x : jax.Array
        Shape ``(d,)`` or ``(B, d)``.

    Returns
    -------
    logit : jax.Array
        Shape ``()`` for a single example or ``(B,)`` for a batch.
    """
    return _mlp_from_params(params).apply({'params': params}, x)
