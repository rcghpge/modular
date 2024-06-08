# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s
# REQUIRES: manual
from python import PythonObject
from max.serve.http import PythonServer, PythonService


@value
struct StubService(PythonService):
    fn handle(
        inout self, owned body: PythonObject, owned handler: PythonObject
    ) raises -> None:
        handler.send_response(200)
        handler.end_headers()


def test_python_server() -> None:
    var server = PythonServer.create("0.0.0.0:8000")
    var service = StubService()
    print("Listening on port 8000!")
    server.serve(service)


def main():
    test_python_server()
