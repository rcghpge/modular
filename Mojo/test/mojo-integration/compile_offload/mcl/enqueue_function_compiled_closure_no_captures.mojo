# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen -elaborate %s -o /dev/null

from std.builtin.device_passable import DevicePassable


def call_closure[
    closure_type: DevicePassable & ImplicitlyCopyable & def() -> None
](closure: closure_type):
    closure()


def main():
    def closure():
        pass

    call_closure(closure)
