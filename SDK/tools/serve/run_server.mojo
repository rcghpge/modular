# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys

from max.tensor import TensorSpec

from max.engine import InferenceSession, InputSpec
from max.serve.util.callbacks import NoopServerCallbacks
from max.serve.kserve import FileModel, GRPCServer, MuxInferenceService


fn main() raises:
    var argv = sys.argv()
    var USAGE = String("Usage: mojo ") + argv[
        0
    ] + " [(--name <name> --path <path>)]..."

    var pairs = List[List[String]]()
    var i = 1
    var name_seen = False
    var name: String = ""
    var path: String = ""
    while i < len(argv):
        if not name_seen and argv[i] == "--name":
            name = argv[i + 1]
            name_seen = True
            i += 2
        elif name_seen and argv[i] == "--path":
            path = argv[i + 1]
            pairs.append(List(name, path))
            name_seen = False
            i += 2
        else:
            print(USAGE)
            raise Error("Failed to parse command-line options")

    if name_seen:
        print(USAGE)
        raise Error("Name argument specified without path!")
    if name == "" or path == "":
        print(USAGE)
        raise Error("No model or path specified!")
    var models = List[FileModel]()
    for pair in pairs:
        models.append(FileModel(pair[][0], "0", pair[][1], None))

    var session = InferenceSession()
    var service = MuxInferenceService(models, session)
    var server = GRPCServer.create("0.0.0.0:8000", session)

    print("Listening on port 8000!")
    server.serve(service)
