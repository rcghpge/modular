# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys

from max.engine import InferenceSession
from max.serve.service import (
    FileModel,
    InferenceRequest,
    InferenceResponse,
)
from max.serve.kserve.callbacks import NoopServerCallbacks
from max.serve.kserve.server import (
    GRPCInferenceServer,
    MuxInferenceService,
)


fn main() raises:
    var USAGE = "Usage: run_sync_server.mojo [(--name <name> --path <path>)]..."
    var argv = sys.argv()

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
        raise Error("Name argument specified without path!")

    var models = List[FileModel]()
    for pair in pairs:
        models.append(FileModel(pair[][0], "0", pair[][1]))

    var session = InferenceSession()
    var service = MuxInferenceService(models, session)
    var server = GRPCInferenceServer.create("0.0.0.0:8000", session)

    print("Listening on port 8000!")
    service.init(server)
    server.serve(service)
