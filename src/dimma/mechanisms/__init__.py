"""One-shot differentially private mechanisms.

This package is the seam between the DP-agnostic geometry/noise primitives in
:mod:`dimma.core` and the higher-level training loops in
:mod:`dimma.algorithms`:

- :mod:`dimma.core` composes geometry and noise but **makes no DP claims** —
  it does not know what was clipped, or how a scale was calibrated.
- :mod:`dimma.algorithms` implements iterative *training loops* (SpiderBoost),
  not standalone primitives.
- :mod:`dimma.mechanisms` (this package) holds **self-contained, one-shot
  DP-claiming primitives**: a single call that takes privacy parameters, adds
  calibrated noise, and returns a private release. A future DP-SGD would call
  one of these once per step.

The first inhabitant is the projection mechanism of Ghazi et al. (2024),
Algorithm 1.
"""

from dimma.mechanisms.projection import projection_mechanism

__all__ = [
    "projection_mechanism",
]
