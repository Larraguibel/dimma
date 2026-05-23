"""Subsampling primitives for DP optimization.

Currently contains Poisson subsampling in two flavors:
- ``poisson_subsample``: strict; raises on oversize batches.
- ``poisson_subsample_truncated``: deterministic truncation on oversize
  batches. Modified mechanism, heuristic accountant.

The two are intentionally distinct so call sites are greppable. Each
has a matching accountant in ``dimma.accounting.sampling``.
"""

from dimma.core.sampling.poisson import (
    poisson_padded_batch_size,
    poisson_subsample,
    poisson_subsample_truncated,
)

__all__ = [
    "poisson_padded_batch_size",
    "poisson_subsample",
    "poisson_subsample_truncated",
]
