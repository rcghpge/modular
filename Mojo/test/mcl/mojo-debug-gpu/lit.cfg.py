# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
import sys

cwd = os.path.dirname(__file__)
sys.path.append(cwd)
import cuda_gdb_runner

# name: The name of this test suite.
config.name = "mojo-debug-gpu"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join("Mojo", "test", "mcl", "mojo-debug-gpu")


# The following enables cuda-gdb tests. It doesn't affect tests that don't use
# cuda-gdb.
config.test_format = cuda_gdb_runner.CudaGDBTest(config=config)

# Handle the very rare case of cuda-gdb getting incomplete debug-info from cuda
# driver.
config.test_retry_attempts = 3

config.available_features.add("debugging-test")
