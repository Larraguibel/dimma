"""Private SpiderBoost (Arora et al., ICML 2023)."""

from dimma.algorithms.spiderboost.kernels import (
    StepOutput,
    make_anchor_step,
    make_variation_step,
    sgd_update,
)
from dimma.algorithms.spiderboost.train import (
    train,
    TrainConfig,
    TrainHistory,
    TrainResult,
    StepInfo,
)
from dimma.algorithms.spiderboost.config import resolve_config

__all__ = [
    "StepOutput",
    "make_anchor_step",
    "make_variation_step",
    "sgd_update",
    "train",
    "TrainConfig",
    "TrainHistory",
    "TrainResult",
    "StepInfo",
    "resolve_config",
]
