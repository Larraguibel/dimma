import shutil
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dimma.datasets import load_criteo
from dimma.datasets._cache import get_cache_dir


def _criteo_cache_path() -> Path:
    return get_cache_dir("datasets") / "criteo_1M.parquet"


requires_criteo = pytest.mark.skipif(
    not _criteo_cache_path().exists(),
    reason="Criteo parquet not in cache; "
           "run `python3 -c 'from dimma.datasets import load_criteo; load_criteo()'` first",
)


def test_download_false_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_criteo(root=tmp_path, download=False)


@requires_criteo
def test_integer_mode_shapes():
    split = load_criteo(features="integer")
    assert split.x_train.shape[1] == 13
    assert split.x_train.dtype == jnp.float32
    assert split.metadata["features"] == "integer"
    assert "feature_means" in split.metadata
    assert "feature_stds" in split.metadata
    assert len(split.metadata["feature_means"]) == 13
    assert len(split.metadata["feature_stds"]) == 13


@requires_criteo
def test_all_mode_shapes():
    split = load_criteo(features="all")
    assert split.x_train.shape[1] == 39
    assert split.metadata["features"] == "all"
    assert "int_cols" in split.metadata
    assert "cat_cols" in split.metadata
    assert len(split.metadata["int_cols"]) == 13
    assert len(split.metadata["cat_cols"]) == 26


@requires_criteo
def test_integer_standardization():
    split = load_criteo(features="integer")
    means = np.asarray(split.x_train).mean(axis=0)
    stds = np.asarray(split.x_train).std(axis=0)
    assert np.allclose(means, 0.0, atol=1e-2)
    assert np.allclose(stds, 1.0, atol=1e-2)


@requires_criteo
def test_all_mode_preserves_raw():
    split = load_criteo(features="all")
    x = np.asarray(split.x_train)
    # Categorical cols (last 26): hashed IDs — wide value range.
    cat_block = x[:, 13:]
    assert np.nanmax(cat_block) > 1000
    # Integer cols: in "all" mode we must NOT apply log1p/standardization.
    # The cached parquet happens to have no NaNs and already non-negative
    # values, so we verify passthrough by reloading the parquet directly
    # and comparing rows.
    import pandas as pd
    parquet = _criteo_cache_path()
    raw = pd.read_parquet(parquet, columns=[f"I{i}" for i in range(1, 14)])
    # The "all"-mode train slice was selected via the same seed/permutation
    # as `dimma.datasets.load_criteo`; reconstruct it.
    rng = np.random.default_rng(0)
    n = len(raw)
    perm = rng.permutation(n)
    n_test = int(round(n * 0.2))
    train_idx = perm[n_test:]
    raw_train_int = raw.iloc[train_idx].to_numpy(dtype=np.float32)
    assert np.array_equal(x[:, :13], raw_train_int)


@requires_criteo
def test_regression_against_source():
    SOURCE_ROOT = Path(__file__).parent.parent.parent / "private_spider_boost_criteo"
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    from src.criteo_loader import load_criteo as src_load

    parquet = _criteo_cache_path()
    src_split = src_load(parquet, test_fraction=0.2, seed=0)
    dimma_split = load_criteo(
        features="integer", root=parquet.parent,
        test_fraction=0.2, seed=0,
    )
    for name in ("x_train", "y_train", "x_test", "y_test"):
        a = np.asarray(getattr(src_split, name))
        b = np.asarray(getattr(dimma_split, name))
        assert np.array_equal(a, b), f"{name} differs"


@requires_criteo
def test_license_notice_on_fresh_download(tmp_path, capsys, monkeypatch):
    # Reset emitted keys so the notice fires this run.
    from dimma.datasets import _attribution
    monkeypatch.setattr(_attribution, "_emitted_keys", set())
    # Copy cached file to tmp_path so we don't actually re-download.
    src = _criteo_cache_path()
    dst = tmp_path / "criteo_1M.parquet"
    # Force the "fresh download" path: leave tmp_path empty, but stub the
    # download to copy from cache.
    from dimma.datasets import criteo as criteo_mod

    def fake_download(url, destination, expected_sha256, chunk_size=1024*1024):
        shutil.copyfile(src, destination)
        return destination

    monkeypatch.setattr(criteo_mod, "download_with_checksum", fake_download)
    load_criteo(root=tmp_path)
    captured = capsys.readouterr()
    assert "CC-BY-NC-SA 4.0" in captured.err
    assert "Criteo Labs" in captured.err


@requires_criteo
def test_license_notice_not_on_cached_read(capsys, monkeypatch):
    # Pretend the notice already fired in this process.
    from dimma.datasets import _attribution
    monkeypatch.setattr(_attribution, "_emitted_keys", {"criteo_1M"})
    load_criteo(features="integer")
    captured = capsys.readouterr()
    assert "CC-BY-NC-SA 4.0" not in captured.err
