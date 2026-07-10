"""Hashed logistic regression — a sparse-gradient reference model.

This is a :term:`reference model` (see ``CONTEXT.md``): a concrete network the
library ships so researchers have a testing model in hand. Like the shipped
:class:`~dimma.models.MLP`, it is **not** part of the architecture-agnostic
algorithm — ``dimma.train`` never depends on it. Model code lives here under
``dimma.models``; it must never appear under ``dimma.algorithms``.

Why a hashed *linear* model
---------------------------
The point of this model is a **sparse per-sample gradient**. Categorical
features (e.g. the 26 Criteo ``C`` columns) are hashed into per-field bucket
tables; each example touches exactly one bucket per field. A single linear
layer means the whole per-sample gradient is

.. math::

    \\nabla_\\theta \\ell = (\\sigma(\\text{logit}) - y) \\cdot \\phi(x)

where ``phi(x)`` is the sparse feature map: one one-hot per categorical field
plus the dense integer features and the bias. The gradient therefore has at
most ``num_fields + num_dense + 1`` nonzeros per sample — regardless of the
table size ``num_fields * num_buckets``. Any *dense* head layer (e.g. Embed +
MLP) would make the global per-sample gradient dense and destroy this
structure, which is exactly what the ``l_1``-ball projection mechanism
(Ghazi et al. 2024) needs to exploit.

Parameter pytree
----------------
``{"table": (num_fields*num_buckets,), "w_dense": (num_dense,), "b": ()}`` — a
plain JAX pytree, fully compatible with ``jax.vmap`` / ``jax.grad`` /
``jax.jit`` and with the per-sample gradient pattern used by ``dimma.train``.

Feature layout
--------------
``forward`` expects a single feature vector ``x`` whose first ``num_dense``
entries are dense (real-valued) features and whose remaining ``num_fields``
entries are **global bucket indices** produced by :func:`hash_buckets`, stored
as ``float32``. Splitting the dense/categorical parts and computing the bucket
indices from raw IDs is the caller's job (dataset-specific glue), keeping this
model dataset-agnostic.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing helper (dataset-agnostic, NumPy)
# ---------------------------------------------------------------------------

def hash_buckets(ids, num_buckets: int) -> np.ndarray:
    """Map raw categorical IDs to per-field global bucket indices.

    Each column ``j`` (a categorical field) is hashed into its own contiguous
    block of ``num_buckets`` slots inside a single shared table. Column ``j``
    maps an id to ``j * num_buckets + (id % num_buckets)``, so distinct fields
    never share a table slot — the per-field blocks are disjoint. This keeps
    the per-field gather (and hence the gradient) confined to exactly one slot
    per field.

    This helper is purely a NumPy preprocessing convenience and makes no
    dataset-specific assumptions; dataset glue (integer preprocessing, which
    columns are categorical) lives with the caller, not here.

    Parameters
    ----------
    ids : array-like, shape ``(..., num_fields)``
        Raw categorical IDs. Cast to ``int64`` before the modulo. Negative IDs
        are handled by NumPy's floored modulo (the result is non-negative for a
        positive ``num_buckets``).
    num_buckets : int
        Number of buckets per field.

    Returns
    -------
    indices : np.ndarray, dtype ``float32``, same shape as ``ids``
        Global bucket indices in ``[0, num_fields * num_buckets)``, packed as
        ``float32``. The cast is **exact** for indices ``<= 2**24``; with the
        notebook defaults (``num_fields=26``, ``num_buckets=1024``) the maximum
        index is ``26623``, comfortably exact. Criteo's categorical hashes
        already arrive as ``float32`` (see ``datasets/criteo.py``); that upstream
        cast is lossy for very large source IDs, adding a few extra hash
        collisions — harmless for this reference model.

    Raises
    ------
    ValueError
        If ``num_fields * num_buckets > 2**24``, since the largest global
        index would then exceed the float32-exact range and the cast would
        alias buckets silently. Enforced eagerly on the NumPy side.
    """
    ids = np.asarray(ids)
    num_fields = ids.shape[-1]

    # The float32 pack is exact only for indices <= 2**24 (float32 has a
    # 24-bit mantissa). The largest global index is
    # (num_fields - 1) * num_buckets + (num_buckets - 1) = num_fields*num_buckets - 1,
    # so requiring num_fields * num_buckets <= 2**24 keeps every index exact.
    # Beyond that the cast silently aliases buckets into wrong slots and the
    # gathered gradients are plausible-but-wrong — fail eagerly instead.
    max_index_count = num_fields * int(num_buckets)
    if max_index_count > 2**24:
        raise ValueError(
            f"num_fields * num_buckets = {num_fields} * {int(num_buckets)} = "
            f"{max_index_count} exceeds 2**24 = {2**24}; float32 bucket "
            "indices would alias silently. Reduce num_buckets (or num_fields) "
            "so the table has at most 2**24 slots."
        )

    offsets = np.arange(num_fields, dtype=np.int64) * int(num_buckets)
    buckets = (ids.astype(np.int64) % int(num_buckets)) + offsets
    return buckets.astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_params(
    key: jax.Array,
    num_dense: int,
    num_fields: int,
    num_buckets: int,
) -> dict:
    """Initialise hashed-logistic-regression parameters.

    Returns the param pytree
    ``{"table": (num_fields*num_buckets,), "w_dense": (num_dense,), "b": ()}``,
    a plain JAX pytree compatible with ``jax.grad`` / ``jax.vmap`` / ``jax.jit``.

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    num_dense : int
        Number of dense (real-valued) features.
    num_fields : int
        Number of categorical fields (each hashed into ``num_buckets`` slots).
    num_buckets : int
        Number of buckets per categorical field.
    """
    k_table, k_dense = jax.random.split(key)
    table = jax.random.normal(k_table, (num_fields * num_buckets,)) * 0.01
    w_dense = jax.random.normal(k_dense, (num_dense,)) * 0.01
    b = jnp.array(0.0)
    return {"table": table, "w_dense": w_dense, "b": b}


def forward(params: dict, x: jax.Array) -> jax.Array:
    """Compute the logit for a **single** example. ``vmap`` for a batch.

    The first ``num_dense = w_dense.shape[0]`` entries of ``x`` are dense
    features; the remaining entries are global bucket indices (as produced by
    :func:`hash_buckets`, stored as ``float32``). The logit is

    ``dot(w_dense, dense) + sum_j table[idx_j] + b``.

    Because each field's index lands in a disjoint table block, the gather
    ``table[idx]`` (and its transpose, the gradient scatter) touches exactly
    ``num_fields`` distinct slots — this is what keeps the gradient sparse.

    Parameters
    ----------
    params : dict
        Param pytree as returned by :func:`init_params`.
    x : jax.Array, shape ``(num_dense + num_fields,)``
        A single feature vector: dense features followed by float-encoded
        global bucket indices.

    Returns
    -------
    logit : jax.Array, shape ``()``
        The scalar logit for this example.
    """
    table = params["table"]
    w_dense = params["w_dense"]
    b = params["b"]

    num_dense = w_dense.shape[0]
    dense = x[:num_dense]
    idx = x[num_dense:].astype(jnp.int32)

    return jnp.dot(w_dense, dense) + jnp.sum(table[idx]) + b
