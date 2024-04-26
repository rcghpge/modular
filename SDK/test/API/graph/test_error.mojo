# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s

import testing

from max.graph.error import error


def test_error():
    with testing.assert_raises(contains="test_error.mojo:15"):
        raise error("blah")


def main():
    test_error()
