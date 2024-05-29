# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: windows
# REQUIRES: manual
# RUN: %mojo -debug-level full %s

import subprocess
import sys


from algorithm import parallelize
from builtin.io import _printf
from pathlib import cwd


fn main() raises:
    var USAGE = "Usage: test_multi_triton.mojo [-a address] [-n workers] [-r requests]"
    var argv = sys.argv()

    var i = 1
    var address: String = "localhost:8000"
    var num_workers = 4
    var num_requests = 4000
    while i < len(argv):
        if argv[i] == "-a":
            address = str(argv[i + 1]).strip()
            if not address:
                raise Error("Must specify non-empty address")
            i += 2
        elif argv[i] == "-n":
            num_workers = atol(argv[i + 1])
            i += 2
        elif argv[i] == "-r":
            num_requests = atol(argv[i + 1])
            i += 2
        else:
            print(USAGE)
            raise Error("Failed to parse command-line options")

    @always_inline
    @parameter
    fn worker(idx: Int):
        try:
            _printf["[[ Output %d ]] %s\n"](
                idx,
                subprocess.run(
                    "mojo "
                    + str(
                        cwd().joinpath(
                            "SDK/test/serve/test_server_single_triton.mojo"
                        )
                    )
                    + " -a "
                    + str(address)
                    + " -n "
                    + str(num_requests)
                ),
            )
        except e:
            print(e)

    _printf["[ Sending %dx%d requests! ]\n"](num_workers, num_requests)
    parallelize[worker](num_workers)
    _ = address^
