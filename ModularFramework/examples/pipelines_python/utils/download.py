# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Download utilities."""

import os
import urllib.request
from pathlib import Path


def modular_cache_dir() -> Path:
    """Follow the convention for caching downloads."""
    xdg_cache_home = os.getenv("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "modular"
    return Path.home() / ".cache" / "modular"


def _is_url(url: str) -> bool:
    """Returns whether url is a web URL."""
    return url.startswith("http://") or url.startswith("https://")


def download_to_cache(url: str) -> Path:
    """If file doesn't exist download to `.cache` and return path."""
    if not _is_url(url):
        return Path(url)

    cache_path = modular_cache_dir()
    os.makedirs(cache_path, exist_ok=True)
    last_component = url.split("/")[-1]
    destination = cache_path / last_component

    if not destination.is_file():
        tmp_destination = str(destination) + ".tmp"
        urllib.request.urlretrieve(url, tmp_destination)
        # Once finished, mv the file so we hit the cache next time
        os.rename(tmp_destination, destination)

    return destination
