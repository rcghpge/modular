# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import getenv
from sys import external_call
from sys.ffi import RTLD, DLHandle
from memory.unsafe import DTypePointer


struct AIEngine:
    """Represents an instance of Modular AI Engine."""

    """Handle to Modular AI Engine library."""
    var lib: DLHandle

    alias VersionFnName = "M_version"

    fn __init__(inout self) raises:
        let engine_lib = getenv("MODULAR_AI_ENGINE_LIB_PATH")
        if engine_lib == "":
            raise "cannot find ai engine library. Please set MODULAR_AI_ENGINE_LIB_PATH."

        self.lib = DLHandle(engine_lib, RTLD.NOW | RTLD.GLOBAL)

    fn get_version(self) -> String:
        """Returns version of modular AI engine.

        Returns:
            Version as string.
        """
        let version_string = self.lib.get_function[
            fn () -> DTypePointer[DType.int8]
        ](self.VersionFnName)()
        return String(version_string)

    fn __del__(owned self):
        self.lib._del_old()
