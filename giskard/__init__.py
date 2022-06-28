# type: ignore[attr-defined]
"""Inspect your AI models visually, find bugs, give feedback 🕵️‍♀️ 💬"""

import sys

if sys.version_info >= (3, 8):
    from importlib import metadata as importlib_metadata
else:
    import importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__ = get_version()

from .giskard_client import GiskardClient
