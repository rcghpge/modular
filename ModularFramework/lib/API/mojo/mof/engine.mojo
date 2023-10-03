# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import getenv
from sys import external_call
from sys.ffi import DLHandle
from memory.unsafe import DTypePointer
from ._status import Status
from ._utils import *


struct AIEngine:
    """Represents an instance of Modular AI Engine."""

    """Handle to Modular AI Engine library."""
    var lib: DLHandle

    alias VersionFnName = "M_version"

    fn __init__(inout self) raises:
        let engine_lib = getenv("MODULAR_AI_ENGINE_LIB_PATH")
        if engine_lib == "":
            raise "cannot find ai engine library. Please set MODULAR_AI_ENGINE_LIB_PATH."

        self.lib = DLHandle(engine_lib)

    fn get_version(self) -> String:
        """Returns version of modular AI engine.

        Returns:
            Version as string.
        """
        let version = call_dylib_func[CString](self.lib, Self.VersionFnName)
        return version.__str__()

    fn get_status(self) -> Status:
        """Creates a Status object."""
        return Status(self.lib)

    fn __del__(owned self):
        self.lib._del_old()
