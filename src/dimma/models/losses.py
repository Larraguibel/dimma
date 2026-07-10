"""Per-sample and batch BCE losses for the reference :class:`~dimma.models.MLP`.

Sigmoid binary cross-entropy losses paired with the shipped reference model.
The per-sample form matches the ``(params, x_single, y_single) -> scalar``
contract ``dimma.train`` expects for per-sample gradient computation. These are
reference-model conveniences under ``dimma.models``; the algorithm itself takes
an arbitrary per-sample loss and never imports them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dimma.models.mlp import forward
from dimma.models.hashed_logreg import forward as hashed_forward


def _stable_bce(logit: jax.Array, y: jax.Array) -> jax.Array:
    """Numerically-stable sigmoid BCE, elementwise over ``logit``.

    Evaluates ``max(logit, 0) - logit * y + log1p(exp(-|logit|))``, the
    overflow-safe form of ``-[y*log σ(logit) + (1-y)*log(1-σ(logit))]``. Single
    source of truth for the formula shared by the losses below; broadcasts over
    any ``logit`` shape (scalar for the per-sample losses, batched for
    :func:`batch_bce_loss`).
    """
    return jnp.maximum(logit, 0.0) - logit * y + jnp.log1p(jnp.exp(-jnp.abs(logit)))


def per_sample_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Sigmoid BCE loss for a **single** example.

    Signature is ``(params, x_single, y_single) -> scalar``, matching the
    contract expected by ``dimma.train`` for per-sample gradient computation::

        jax.vmap(jax.grad(per_sample_bce_loss), in_axes=(None, 0, 0))

    Parameters
    ----------
    params : dict
        Flax param pytree.
    x : jax.Array, shape (d,)
        Single feature vector.
    y : jax.Array, shape ()
        Binary label in {0., 1.}.
    """
    logit = forward(params, x)
    return _stable_bce(logit, y)


def per_sample_hashed_bce_loss(
    params: dict, x: jax.Array, y: jax.Array
) -> jax.Array:
    """Sigmoid BCE loss for a **single** example, hashed-logreg model.

    Identical numerically-stable BCE formula as :func:`per_sample_bce_loss`,
    but evaluates the logit with :func:`dimma.models.hashed_logreg.forward`.
    The logit map is linear in the (sparse) feature vector, so the per-sample
    gradient is sparse — see ``dimma.models.hashed_logreg`` for why.

    Signature is ``(params, x_single, y_single) -> scalar``, matching the
    contract expected by ``dimma.train`` for per-sample gradient computation::

        jax.vmap(jax.grad(per_sample_hashed_bce_loss), in_axes=(None, 0, 0))

    Parameters
    ----------
    params : dict
        Hashed-logreg param pytree.
    x : jax.Array, shape (num_dense + num_fields,)
        Single feature vector: dense features followed by float-encoded global
        bucket indices (see :func:`dimma.models.hashed_logreg.hash_buckets`).
    y : jax.Array, shape ()
        Binary label in {0., 1.}.
    """
    logit = hashed_forward(params, x)
    return _stable_bce(logit, y)


def batch_bce_loss(params: dict, x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean BCE loss over a batch (used internally by ``train_spider``).

    Parameters
    ----------
    params : dict
        Flax param pytree.
    x : jax.Array, shape (B, d)
        Batch of feature vectors.
    y : jax.Array, shape (B,)
        Batch of binary labels.
    """
    logits = forward(params, x)
    return jnp.mean(_stable_bce(logits, y))
