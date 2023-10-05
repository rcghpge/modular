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
from .session import InferenceSession
from ._tensor_spec import MOFTensorSpec
from tensor import TensorSpec


struct AIEngine:
    """Represents an instance of Modular AI Engine."""

    """Handle to Modular AI Engine library."""
    var lib: DLHandle
    var can_close_lib: Bool

    alias VersionFnName = "M_version"

    fn __init__(inout self) raises:
        let engine_lib = getenv("MODULAR_AI_ENGINE_LIB_PATH")
        if engine_lib == "":
            raise "cannot find ai engine library. Please set MODULAR_AI_ENGINE_LIB_PATH."

        self.lib = DLHandle(engine_lib)
        self.can_close_lib = True

    fn __moveinit__(inout self, owned existing: Self):
        self.lib = existing.lib
        self.can_close_lib = exchange[Bool](existing.can_close_lib, False)

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

    fn __enter__(owned self) -> Self:
        return self ^

    fn get_session(self) raises -> InferenceSession:
        """
        Gets an InferenceSession Object.
        """
        return InferenceSession(self.lib)

    fn get_as_engine_tensor_spec(
        self, name: StringRef, spec: TensorSpec
    ) -> MOFTensorSpec:
        return MOFTensorSpec(name, spec, self.lib)

    fn __del__(owned self):
        if self.can_close_lib:
            self.lib._del_old()
