"""dimma — JAX-based library of differentially private optimization algorithms."""

from importlib.metadata import version

__version__ = version("dimma")

from dimma.algorithms.spiderboost import (
    train,
    TrainConfig,
    TrainHistory,
    TrainResult,
    StepInfo,
    resolve_config,
)
from dimma.accounting import (
    NoiseScales,
    compute_noise_scales,
)

__all__ = [
    "__version__",
    "train",
    "resolve_config",
    "TrainConfig",
    "TrainHistory",
    "TrainResult",
    "StepInfo",
    "NoiseScales",
    "compute_noise_scales",
]
