# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Execution Configurations for CPUs

For now, CPUs are treated as a single target, but this may change in the future.
"""
from .. import ExecutionConfig
from .cpu_execution_config import CPUExecutionConfiguration
from std.collections.type_dict import TypeDict
