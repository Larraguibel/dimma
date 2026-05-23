"""Device selection helper.

Accepts a friendly string (``"cpu"``, ``"cuda"``, ``"gpu"``) and returns the
corresponding ``jax.Device``. Raises a clear error when CUDA is requested
but no GPU backend is available.

Extracted from `private_spider_boost_criteo/src/device.py` without
modification.
"""

from __future__ import annotations

import jax


_ALIASES = {"cpu": "cpu", "gpu": "gpu", "cuda": "gpu"}


def resolve_device(name: str) -> jax.Device:
    """Return the first ``jax.Device`` matching ``name``.

    Parameters
    ----------
    name : str
        ``"cpu"``, ``"gpu"``, or ``"cuda"`` (case-insensitive). ``"cuda"`` is
        an alias for ``"gpu"`` — JAX's backend name is ``gpu`` regardless of
        the underlying CUDA/ROCm/Metal driver.

    Returns
    -------
    jax.Device

    Raises
    ------
    ValueError
        If ``name`` is unknown.
    RuntimeError
        If the requested backend has no available devices (e.g. ``"cuda"``
        on a machine without a CUDA-enabled JAX build).
    """
    key = name.lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown device {name!r}. Expected one of: cpu, gpu, cuda."
        )
    backend = _ALIASES[key]
    try:
        devices = jax.devices(backend)
    except RuntimeError as e:
        raise RuntimeError(
            f"Requested device {name!r} but JAX backend {backend!r} is "
            f"unavailable. Install a CUDA-enabled jaxlib to use the GPU, or "
            f"set device='cpu'. Original error: {e}"
        ) from e
    if not devices:
        raise RuntimeError(f"No devices found for backend {backend!r}.")
    return devices[0]
