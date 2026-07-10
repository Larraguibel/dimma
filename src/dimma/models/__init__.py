"""Reference models shipped with dimma.

A :term:`reference model` is a concrete network the library ships so researchers
have a testing model in hand (``CONTEXT.md`` → "Reference model"). These are
distinct from the architecture-agnostic algorithm in ``dimma.algorithms``, which
never depends on a model. Model code belongs here, never under
``dimma.algorithms`` (see ``docs/adr/0002-thm-b3-config-resolver.md`` and
``CLAUDE.md``).

Two reference models ship today:

- :class:`MLP` (dense gradients) — exports ``init_params`` / ``forward``.
- ``hashed_logreg`` (sparse gradients) — a hashed logistic regression whose
  per-sample gradient is sparse. Its ``init_params`` / ``forward`` collide by
  name with the MLP's, so they are exposed via the ``hashed_logreg`` module and
  the aliases ``hashed_init_params`` / ``hashed_forward`` / ``hash_buckets``;
  the bare ``init_params`` / ``forward`` names stay bound to the MLP.
"""

from dimma.models import hashed_logreg
from dimma.models.mlp import (
    MLP,
    init_params,
    forward,
    _mlp_from_params,
)
from dimma.models.hashed_logreg import (
    hash_buckets,
    init_params as hashed_init_params,
    forward as hashed_forward,
)
from dimma.models.losses import (
    per_sample_bce_loss,
    per_sample_hashed_bce_loss,
    batch_bce_loss,
)

__all__ = [
    "MLP",
    "init_params",
    "forward",
    "hashed_logreg",
    "hash_buckets",
    "hashed_init_params",
    "hashed_forward",
    "per_sample_bce_loss",
    "per_sample_hashed_bce_loss",
    "batch_bce_loss",
]
