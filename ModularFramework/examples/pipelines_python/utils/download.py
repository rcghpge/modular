# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Download utilities."""

import hashlib
from warnings import warn
import os
import urllib.request
import requests
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


def download_to_cache(url: str, verify: bool) -> Path:
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
        if _check_HF_sha(url, tmp_destination):
            os.rename(tmp_destination, destination)

    else:
        if verify and not _check_HF_sha(url, destination):
            raise RuntimeError(
                "Local weights failed checksum and --verify flag was passed"
            )

    return destination


def _check_HF_sha(url: str, local_fname: Path) -> bool:
    # /raw/ in huggingface file URLs resolves to the github LFS pointer (rather than forwarding to resolve the actual file itself)
    r = requests.get(url.replace("resolve", "raw"))

    # Github LFS pointer should have a line with the sha256 hash in it; find it
    http_response_lines = r.text.split("\n")
    lfs_ptr_metadata = {
        k.strip(): v.strip()
        for k, v in [
            line.split(" ") for line in http_response_lines if len(line) > 0
        ]
    }
    assert lfs_ptr_metadata["version"] == "https://git-lfs.github.com/spec/v1"
    assert lfs_ptr_metadata["oid"].startswith("sha256")

    # Parse out the
    expected_sha = lfs_ptr_metadata["oid"].split(":")[1]

    # Compute the sha of the local weights file and compare them
    actual_sha = _get_sha_sum(local_fname)
    if actual_sha != expected_sha:
        warn(
            f"SHA mismatch for file {local_fname}. Expected {expected_sha} but"
            f" local file sha is {actual_sha}"
        )

    return actual_sha == expected_sha


def _get_sha_sum(local_fname: Path) -> str:
    with open(local_fname, "rb", buffering=2) as f:
        buffer = f.read()
        return hashlib.sha256(buffer).hexdigest()
