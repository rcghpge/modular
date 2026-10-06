# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# This test verifies that the following data tables of Mojo accelerator values
# are in sync and don't drift apart:
#
# 1. C++ implementation of `--print-supported-accelerators`
# 2. Mojo `_all_target_accelerator_values` list in `std._gpu.host.info`
#
# Both should produce a list of the same target accelerator values.
#
# Step 1: Get architecture IDs from C++ --print-supported-accelerators output.
# Extract lines between # markers, keep only indented non-<arch> lines, and
# extract the first field.
# RUN: %mojo-build --print-supported-accelerators 2>&1 \
# RUN:   | sed -n '/^#$/,/^#$/ p' \
# RUN:   | grep '^  ' \
#      Remove e.g. `nvidia:<arch>` example values
# RUN:   | grep -v '<arch>' \
# RUN:   | awk '{print $1}' \
# RUN:   | sort -u > %t.cpp_targets.txt

# Step 2: Get architecture IDs from Mojo _all_target_accelerator_values.
# RUN: %mojo %S/inputs/print_all_targets_helper.mojo 2>&1 \
# RUN:   | sort -u > %t.mojo_targets.txt

# Step 3: Compare the two lists. They should be identical.
# RUN: diff %t.cpp_targets.txt %t.mojo_targets.txt

# If the diff fails, the test fails, indicating the lists are out of sync.


def main():
    pass
