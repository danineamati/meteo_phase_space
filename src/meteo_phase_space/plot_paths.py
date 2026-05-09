"""Filesystem-safe paths for plots grouped by location label."""

from __future__ import annotations

import re
from pathlib import Path


def location_slug(label: str) -> str:
    """Turn a human-readable location label into a single directory-safe token."""
    s = label.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.ASCII)
    s = re.sub(r"[-\s]+", "_", s)
    s = s.strip("_") or "location"
    return s[:200]


def plots_subdir(root: Path | str, label: str) -> Path:
    """Return ``{root}/{location_slug(label)}/``, creating intermediate dirs as needed."""
    path = Path(root) / location_slug(label)
    path.mkdir(parents=True, exist_ok=True)
    return path
