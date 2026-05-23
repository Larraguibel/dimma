"""Criteo 1M sample loader.

Provides ``load_criteo``: downloads the Criteo 1M parquet sample from
Hugging Face on first call, then returns a ``TabularSplit``.

License
-------
The Criteo dataset is licensed CC-BY-NC-SA 4.0. ``load_criteo`` prints
an attribution and license notice once per process on first call.
``dimma`` itself is not CC-BY-NC-SA; only the data carries this
restriction. See ``dimma/README.md`` for details on dataset licensing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from dimma.datasets._attribution import emit_once
from dimma.datasets._cache import get_cache_dir
from dimma.datasets._download import download_with_checksum
from dimma.datasets.base import TabularSplit, arrays_to_split


_CRITEO_URL = (
    "https://huggingface.co/datasets/eldieguinpo/criteo-1M/"
    "resolve/main/criteo_1M.parquet"
)
_CRITEO_SHA256 = (
    "0b468148aecf6fa9464def4f2ad075b8843874f3469239afd3504602579f767a"
)
_CRITEO_FILENAME = "criteo_1M.parquet"

_INT_COLS: list[str] = [f"I{i}" for i in range(1, 14)]
_CAT_COLS: list[str] = [f"C{i}" for i in range(1, 27)]
_LABEL_COL: str = "label"

_LICENSE_NOTICE = (
    "Downloaded Criteo 1M sample (CC-BY-NC-SA 4.0).\n"
    "Original data: Criteo Labs (https://ailab.criteo.com).\n"
    "Non-commercial use only. "
    "Derivative works must be shared under the same license."
)


def load_criteo(
    features: Literal["integer", "all"] = "integer",
    root: str | Path | None = None,
    download: bool = True,
    test_fraction: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
) -> TabularSplit:
    """Load the Criteo 1M sample.

    On first call, downloads ``criteo_1M.parquet`` from Hugging Face
    (https://huggingface.co/datasets/eldieguinpo/criteo-1M) to the
    cache directory. Subsequent calls reuse the cached file.

    Parameters
    ----------
    features : {"integer", "all"}, default "integer"
        - ``"integer"``: returns only the 13 integer features
          ``I1..I13``, with log1p + per-feature standardization. NaN
          values are filled with the per-column median computed on the
          training split (no leakage from the test split).
        - ``"all"``: returns ALL 39 raw features (13 integer + 26
          categorical). **No preprocessing**: no log1p, no
          standardization, no hashing, no encoding. NaN values in
          integer columns are left as NaN. The caller is responsible
          for all downstream preprocessing. Categorical columns are
          int64 hashed IDs from the source parquet, cast to float32
          (which loses precision for very large IDs).
    root : str | Path | None, default None
        Cache directory. If ``None``, uses
        ``dimma.datasets._cache.get_cache_dir("datasets")``.
    download : bool, default True
        If ``True``, download the file if missing. If ``False`` and the
        file is missing, raise ``FileNotFoundError``.
    test_fraction : float, default 0.2
        Fraction of rows held out for testing.
    seed : int, default 0
        RNG seed for the train/test split.
    device : str, default "cpu"
        Target JAX device.

    Returns
    -------
    TabularSplit
        With ``metadata`` containing:
        - ``"features"``: the mode ("integer" or "all").
        - ``"license"``: ``"CC-BY-NC-SA 4.0"``.
        - ``"source"``: the HF URL.
        - For ``"integer"`` mode also: ``"feature_means"``,
          ``"feature_stds"`` (numpy arrays of shape ``(13,)``).
        - For ``"all"`` mode also: ``"int_cols"``, ``"cat_cols"`` (lists
          of column names).

    Raises
    ------
    FileNotFoundError
        If ``download=False`` and the file is not in the cache.
    RuntimeError
        If the downloaded file's SHA256 does not match the expected
        digest.

    Notes
    -----
    The library does not enforce the CC-BY-NC-SA 4.0 license; it is the
    caller's responsibility to use the data accordingly. A one-time
    license notice is printed to stderr on first download per process.
    """
    if root is None:
        root = get_cache_dir("datasets")
    else:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
    parquet_path = root / _CRITEO_FILENAME

    if not parquet_path.exists():
        if not download:
            raise FileNotFoundError(
                f"Criteo parquet not found at {parquet_path} and "
                f"download=False. Pass download=True or place the file "
                f"manually."
            )
        download_with_checksum(_CRITEO_URL, parquet_path, _CRITEO_SHA256)
        emit_once("criteo_1M", _LICENSE_NOTICE)

    if features == "integer":
        cols = _INT_COLS + [_LABEL_COL]
    elif features == "all":
        cols = _INT_COLS + _CAT_COLS + [_LABEL_COL]
    else:
        raise ValueError(
            f"Unknown features mode: {features!r}. Expected 'integer' or 'all'."
        )
    df = pd.read_parquet(parquet_path, columns=cols)

    rng = np.random.default_rng(seed)
    n = len(df)
    perm = rng.permutation(n)
    n_test = int(round(n * test_fraction))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    base_meta = {
        "features": features,
        "license": "CC-BY-NC-SA 4.0",
        "source": _CRITEO_URL,
    }

    if features == "integer":
        x_train_df = df.iloc[train_idx][_INT_COLS]
        x_test_df = df.iloc[test_idx][_INT_COLS]
        y_train = df.iloc[train_idx][_LABEL_COL].to_numpy(dtype=np.float32)
        y_test = df.iloc[test_idx][_LABEL_COL].to_numpy(dtype=np.float32)

        medians = x_train_df.median(numeric_only=True)
        x_train_np = x_train_df.fillna(medians).to_numpy(dtype=np.float32)
        x_test_np = x_test_df.fillna(medians).to_numpy(dtype=np.float32)

        x_train_np = np.log1p(np.clip(x_train_np, a_min=0.0, a_max=None))
        x_test_np = np.log1p(np.clip(x_test_np, a_min=0.0, a_max=None))

        means = x_train_np.mean(axis=0)
        stds = x_train_np.std(axis=0)
        stds = np.where(stds < 1e-8, 1.0, stds)

        x_train_np = (x_train_np - means) / stds
        x_test_np = (x_test_np - means) / stds

        metadata = {
            **base_meta,
            "feature_means": means,
            "feature_stds": stds,
        }
    else:  # "all"
        all_cols = _INT_COLS + _CAT_COLS
        x_train_np = df.iloc[train_idx][all_cols].to_numpy(dtype=np.float32)
        x_test_np = df.iloc[test_idx][all_cols].to_numpy(dtype=np.float32)
        y_train = df.iloc[train_idx][_LABEL_COL].to_numpy(dtype=np.float32)
        y_test = df.iloc[test_idx][_LABEL_COL].to_numpy(dtype=np.float32)

        metadata = {
            **base_meta,
            "int_cols": list(_INT_COLS),
            "cat_cols": list(_CAT_COLS),
        }

    return arrays_to_split(
        x_train_np, y_train, x_test_np, y_test,
        device=device, metadata=metadata,
    )
