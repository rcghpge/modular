# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s

import testing

from max.graph.error import error, format_system_stack


def test_error():
    with testing.assert_raises(contains="test_error.mojo:15"):
        raise error("blah")
    with testing.assert_raises(contains="System stack:"):
        raise error("blah")


def test_format_system_stack():
    stack_trace = format_system_stack()
    testing.assert_true("System stack:" in stack_trace)


def main():
    test_error()
    test_format_system_stack()
