"""One-time license/attribution notice printer."""

from __future__ import annotations

import sys
import threading

_emitted_keys: set[str] = set()
_emit_lock = threading.Lock()


def emit_once(key: str, message: str) -> None:
    """Print ``message`` to stderr exactly once per process per ``key``.

    Uses a module-level set of seen keys to deduplicate. Thread-safe
    via a lock.

    Parameters
    ----------
    key : str
        A unique identifier for the message (typically the dataset
        name). The same key never prints twice in one process.
    message : str
        The message to print. Printed verbatim to ``sys.stderr``,
        followed by a newline.
    """
    with _emit_lock:
        if key in _emitted_keys:
            return
        _emitted_keys.add(key)
        print(message, file=sys.stderr)
