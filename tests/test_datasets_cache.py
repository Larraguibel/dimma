from pathlib import Path

from dimma.datasets._cache import get_cache_dir


def test_default_exists():
    p = get_cache_dir()
    assert isinstance(p, Path)
    assert p.exists()


def test_subdir_created():
    p = get_cache_dir("datasets")
    assert p.exists()
    assert p.name == "datasets"


def test_dimma_home_override(tmp_path, monkeypatch):
    monkeypatch.setenv("DIMMA_HOME", str(tmp_path))
    p = get_cache_dir()
    assert p.exists()
    assert str(p).startswith(str(tmp_path.resolve()))


def test_idempotent():
    p1 = get_cache_dir("datasets")
    p2 = get_cache_dir("datasets")
    assert p1 == p2
