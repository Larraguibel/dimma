"""Core DP-optimization primitives: pytree utilities, clipping, noise."""

from dimma.core.pytree import (
    pytree_global_norm,
    pytree_sum_over_batch,
    pytree_scale,
    pytree_add,
    pytree_sub,
    pytree_zeros_like,
)
from dimma.core.clipping import (
    per_sample_norms,
    per_sample_clip,
    per_sample_apply_mask,
)
from dimma.core.noise import add_pytree_gaussian_noise

__all__ = [
    "pytree_global_norm",
    "pytree_sum_over_batch",
    "pytree_scale",
    "pytree_add",
    "pytree_sub",
    "pytree_zeros_like",
    "per_sample_norms",
    "per_sample_clip",
    "per_sample_apply_mask",
    "add_pytree_gaussian_noise",
]
