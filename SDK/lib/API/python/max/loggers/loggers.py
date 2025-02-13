# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Package for initializing loggers across max and setting up the default configuration"""

import logging

from max.serve.config import Settings
from max.serve.telemetry import common


def _init_max_logger():
    """Configures default logging settings for the max package"""
    server_settings = Settings()
    common.configure_metrics(server_settings)
    common.configure_logging(server_settings)


_init_max_logger()


# Getting max loggers through get_logger is important to ensure this file is imported.
# Otherwise, the max default logging configuration will not get setup.
def get_logger(name: str) -> logging.Logger:
    if not name.startswith("max."):
        msg = f"Logger ({name}) must start with `max.` prefix"
        raise ValueError(msg)

    return logging.getLogger(name)
