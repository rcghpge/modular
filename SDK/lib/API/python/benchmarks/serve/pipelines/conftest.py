#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""
Configuration for pytest-benchmark in the pipelines benchmark suite.

This conftest.py file configures pytest-benchmark options to ensure proper
ordering and display of benchmark results for parameterized tests.
"""


def pytest_configure(config):  # noqa: ANN001
    """Configure pytest-benchmark options for optimal result display."""

    # Sort by test name to respect parameter ID ordering
    # This ensures that parameterized tests appear in the order defined in the IDs
    config.option.benchmark_sort = "name"

    # Group by min_tokens parameter to show token size progression clearly
    # This groups results by token size ranges for easier comparison
    config.option.benchmark_group_by = "param:min_tokens"

    # Customize columns shown in results table for comprehensive analysis
    config.option.benchmark_columns = [
        "min",  # Minimum execution time
        "max",  # Maximum execution time
        "mean",  # Mean execution time
        "stddev",  # Standard deviation
        "median",  # Median execution time
        "rounds",  # Number of benchmark rounds
        "iterations",  # Number of iterations per round
    ]

    # Enable warmup for more stable and accurate benchmark results
    config.option.benchmark_warmup = True
    config.option.benchmark_warmup_iterations = 2

    # Optional: Uncomment to save results to JSON for further analysis
    # config.option.benchmark_json = 'benchmark_results.json'

    # Optional: Uncomment to save detailed stats
    # config.option.benchmark_save = 'benchmark_stats'

    print(
        "✓ pytest-benchmark configured: sorting by name, grouping by min_tokens"
    )
