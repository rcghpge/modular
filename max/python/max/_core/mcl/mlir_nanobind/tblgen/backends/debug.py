# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Debug backend."""

from __future__ import annotations

from collections.abc import Iterable

from ..backend import Backend
from ..types import Entry


class DebugBackend(Backend):
    def generate_all(self, entries: Iterable[Entry]):  # noqa: ANN201
        for entry in entries:
            yield f"/// {entry.name}"
            yield repr(entry)
            yield ""
