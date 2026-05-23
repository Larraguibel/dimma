import hashlib
import io
from pathlib import Path
from unittest import mock

import pytest

from dimma.datasets._download import download_with_checksum


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_returns_existing_matching_file(tmp_path):
    dest = tmp_path / "f.bin"
    data = b"hello world"
    dest.write_bytes(data)
    out = download_with_checksum(
        "http://invalid.localhost/never-fetched",
        dest, _sha256(data),
    )
    assert out == dest
    assert out.read_bytes() == data


def test_existing_mismatched_redownloaded(tmp_path):
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"bad cached")
    good = b"good fresh data"
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    chunks = [good, b""]
    resp.read.side_effect = chunks
    with mock.patch("urllib.request.urlopen", return_value=resp):
        out = download_with_checksum("http://x/y", dest, _sha256(good))
    assert out.read_bytes() == good


def test_sha256_mismatch_raises(tmp_path):
    dest = tmp_path / "f.bin"
    payload = b"actual bytes"
    expected_wrong = _sha256(b"different content")
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    resp.read.side_effect = [payload, b""]
    with mock.patch("urllib.request.urlopen", return_value=resp):
        with pytest.raises(RuntimeError) as exc:
            download_with_checksum("http://x/y", dest, expected_wrong)
    msg = str(exc.value)
    assert "SHA256 mismatch" in msg
    assert expected_wrong in msg
    assert _sha256(payload) in msg
    partial = dest.with_suffix(dest.suffix + ".partial")
    assert partial.exists()


def test_atomic_rename_on_success(tmp_path):
    dest = tmp_path / "f.bin"
    payload = b"good"
    resp = mock.MagicMock()
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    resp.read.side_effect = [payload, b""]
    with mock.patch("urllib.request.urlopen", return_value=resp):
        download_with_checksum("http://x/y", dest, _sha256(payload))
    assert dest.exists()
    partial = dest.with_suffix(dest.suffix + ".partial")
    assert not partial.exists()
