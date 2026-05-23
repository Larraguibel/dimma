"""Dataset-agnostic types and helpers.

These do not depend on any specific dataset. They are shared across all
loaders in ``dimma.datasets``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dimma.utils.device import resolve_device


class TabularSplit(NamedTuple):
    """Train/test split for tabular data.

    Attributes
    ----------
    x_train, y_train, x_test, y_test : jnp.ndarray
        Features and labels on the requested device.
    metadata : dict
        Free-form per-dataset extras (means, stds, vocabulary sizes,
        column names, license info, etc.). Empty dict by default.
    """
    x_train: jnp.ndarray
    y_train: jnp.ndarray
    x_test: jnp.ndarray
    y_test: jnp.ndarray
    metadata: dict


def arrays_to_split(
    x_train_np: np.ndarray,
    y_train_np: np.ndarray,
    x_test_np: np.ndarray,
    y_test_np: np.ndarray,
    device: str = "cpu",
    metadata: dict | None = None,
) -> TabularSplit:
    """Move pre-split NumPy arrays onto ``device`` and build a TabularSplit."""
    dev = resolve_device(device)
    return TabularSplit(
        x_train=jax.device_put(jnp.asarray(x_train_np), dev),
        y_train=jax.device_put(jnp.asarray(y_train_np), dev),
        x_test=jax.device_put(jnp.asarray(x_test_np), dev),
        y_test=jax.device_put(jnp.asarray(y_test_np), dev),
        metadata=metadata if metadata is not None else {},
    )
