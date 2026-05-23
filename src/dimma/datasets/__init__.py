"""Dataset loaders.

Convenience layer for downloading and preprocessing canonical datasets.
The library's algorithms (``dimma.train``, kernels, accountants) do not
depend on this module — it exists only to make benchmarking and
tutorials easier.

Each dataset is loaded by a top-level function (e.g. ``load_criteo``)
that handles cache lookup, download, checksum verification, and
preprocessing. The first time a dataset is downloaded in a process, a
one-line license notice is printed to stderr.

Currently available:
- Criteo 1M (CC-BY-NC-SA 4.0): ``load_criteo``
"""

from dimma.datasets.base import TabularSplit, arrays_to_split
from dimma.datasets.criteo import load_criteo

__all__ = ["TabularSplit", "arrays_to_split", "load_criteo"]
