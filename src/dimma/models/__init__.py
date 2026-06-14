"""Reference models shipped with dimma.

A :term:`reference model` is a concrete network the library ships so researchers
have a testing model in hand (``CONTEXT.md`` → "Reference model"). These are
distinct from the architecture-agnostic algorithm in ``dimma.algorithms``, which
never depends on a model. Model code belongs here, never under
``dimma.algorithms`` (see ``docs/adr/0002-thm-b3-config-resolver.md`` and
``CLAUDE.md``).
"""

from dimma.models.mlp import (
    MLP,
    init_params,
    forward,
    _mlp_from_params,
)
from dimma.models.losses import (
    per_sample_bce_loss,
    batch_bce_loss,
)

__all__ = [
    "MLP",
    "init_params",
    "forward",
    "per_sample_bce_loss",
    "batch_bce_loss",
]
