# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Helper file for testing --print-target-accelerators synchronization.
# This prints the _all_target_accelerators() values accepted by max.gpu.host.info so
# they can be compared against the C++ implementation's output.

from std._gpu.host.info import _all_target_accelerators


def main():
    # Print each normalized value on its own line for set comparison.

    comptime values = _all_target_accelerators()

    comptime for i in range(len(values)):
        comptime value = values[i]

        # TODO(MSTDL-3241): Remove empty target sentinel value.
        comptime if i == 0 and value == "":
            continue

        print(value)
