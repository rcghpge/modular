# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys

from max.engine import InferenceSession
from max.serve.service import ModelInfo
from max.serve.kserve import KServeGRPCInferenceServer


fn main() raises:
    var USAGE = "Usage: run_sync_server.mojo [--name <name>] [--path <path>]"
    var argv = sys.argv()

    var i = 1
    var name: String = ""
    var path: String = ""
    while i < len(argv):
        if argv[i] == "--name":
            name = argv[i + 1]
            i += 2
        elif argv[i] == "--path":
            path = argv[i + 1]
            i += 2
        else:
            print(USAGE)
            raise Error("Failed to parse command-line options")

    if not name:
        raise Error("Cannot specify an empty model name!")
    if not path:
        raise Error("Cannot specify an empty model path!")

    var session = InferenceSession()
    var server = KServeGRPCInferenceServer(
        "0.0.0.0:8000",
        ModelInfo(name, path, 0),
        session^,
    )

    print("Listening on port 8000!")
    server.serve()
