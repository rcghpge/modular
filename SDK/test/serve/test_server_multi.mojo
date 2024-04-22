# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# RUN: %mojo -debug-level full %s

import subprocess

from algorithm import parallelize
from builtin.io import _printf
from pathlib import cwd


fn main() raises:
    alias num_requests = 500
    alias num_workers = 16

    @parameter
    fn worker(idx: Int):
        try:
            var output = subprocess.run(
                "mojo "
                + str(cwd().joinpath("SDK/test/serve/test_server_single.mojo"))
                + " --num-requests "
                + str(num_requests)
            )
            _printf("[[ Worker %d output ]]\n%s...\n", idx, output[:300])
        except e:
            print(e)

    parallelize[worker](num_workers)
