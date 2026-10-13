# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""JSON backend."""

from __future__ import annotations

import json
from collections.abc import Iterable

from ..backend import Backend, indent
from ..types import Entry


class JsonBackend(Backend):
    def generate_all(self, entries: Iterable[Entry]):  # noqa: ANN201
        entries = list(entries)
        yield "["
        for i, entry in enumerate(entries):
            raw = json.dumps(entry.data)
            if i < len(entries) - 1:
                raw += ","
            yield indent(raw, 1)
        yield "]"
