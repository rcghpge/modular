# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Public Python surface for the Rust host/disk tiered KV connector.

``TierConnector`` and ``TierTransfer`` are implemented in Rust and exposed by
the ``kv_tier_connector._kv_tier_connector_pyo3`` extension module. This shim
re-exports them so callers ``from kv_tier_connector import TierConnector``.
"""

from __future__ import annotations

from ._kv_tier_connector_pyo3 import (  # type: ignore[import-not-found]
    TierConnector,
    TierTransfer,
)

__all__ = ["TierConnector", "TierTransfer"]
