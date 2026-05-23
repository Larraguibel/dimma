"""Idempotent download with SHA256 verification."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_checksum(
    url: str,
    destination: Path,
    expected_sha256: str,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file to ``destination``, verifying its SHA256.

    Behavior:
    1. If ``destination`` already exists and its SHA256 matches
       ``expected_sha256``, return it without re-downloading.
    2. If ``destination`` exists but its SHA256 does NOT match, delete
       it and re-download.
    3. Otherwise download from ``url`` to a temporary path
       (``destination.with_suffix('.partial')``), verify the SHA256, and
       atomically rename to ``destination``.

    Raises
    ------
    RuntimeError
        If the downloaded file's SHA256 does not match
        ``expected_sha256``. The partial file is left in place for
        debugging, with a clear message stating the mismatch.

    Parameters
    ----------
    url : str
        HTTPS URL of the file to download.
    destination : Path
        Final on-disk path. Parent directory must already exist.
    expected_sha256 : str
        64-character lowercase hex SHA256 digest. Compared
        case-insensitively.
    chunk_size : int, default 1 MiB
        Streaming chunk size for the download.

    Returns
    -------
    Path
        The verified ``destination`` path.

    Notes
    -----
    Uses ``urllib.request.urlopen`` from the standard library. No
    external dependencies are added. No retry logic; if the network
    fails, the user gets the underlying ``URLError`` and can retry.
    """
    expected = expected_sha256.lower()
    destination = Path(destination)

    if destination.exists():
        if _compute_sha256(destination) == expected:
            return destination
        destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".partial")
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as resp, open(partial, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"Downloaded file SHA256 mismatch for {url}.\n"
            f"Expected: {expected}\n"
            f"Got: {actual}\n"
            f"Partial file kept at: {partial}"
        )
    partial.rename(destination)
    return destination
