# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


from collections.optional import Optional


from utils.variant import Variant

import ._c
import ._c.Diagnostics
import ._c.IR
from ._c.ffi import MLIR_func
from ._c.Support import MlirLogicalResult


@register_passable("trivial")
struct DiagnosticSeverity(Copyable, Movable):
    """Severity level of a diagnostic."""

    alias cType = _c.Diagnostics.MlirDiagnosticSeverity
    var c: Self.cType

    alias ERROR = _c.Diagnostics.MlirDiagnosticError
    alias WARNING = _c.Diagnostics.MlirDiagnosticWarning
    alias NOTE = _c.Diagnostics.MlirDiagnosticNote
    alias REMARK = _c.Diagnostics.MlirDiagnosticRemark

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn __eq__(self, other: Self) -> Bool:
        return self.c.value == other.c.value


struct Diagnostic(Copyable, Movable, Stringable, Writable):
    """An opaque reference to a diagnostic, always owned by the diagnostics engine
    (context). Must not be stored outside of the diagnostic handler."""

    alias cType = _c.Diagnostics.MlirDiagnostic
    var c: Self.cType

    @implicit
    fn __init__(out self, c: Self.cType):
        self.c = c

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        _c.Diagnostics.mlirDiagnosticPrint(writer, self.c)

    fn get_severity(self) -> DiagnosticSeverity:
        return _c.Diagnostics.mlirDiagnosticGetSeverity(self.c)


alias DiagnosticHandlerID = _c.Diagnostics.MlirDiagnosticHandlerID


@fieldwise_init
struct DiagnosticHandler[handler: fn (Diagnostic) -> Bool](Copyable, Movable):
    """Deals with attaching and detaching diagnostic functions to an MLIRContext.

    Parameters:
        handler: A function that handles a given Diagnostic.
    """

    var id: DiagnosticHandlerID
    var ctx: Context

    @staticmethod
    @always_inline
    fn attach(ctx: Context) -> Self:
        fn delete_user_data(data: OpaquePointer):
            pass

        var id = _c.Diagnostics.mlirContextAttachDiagnosticHandler(
            ctx.c,
            Self.c_handler,
            OpaquePointer(),
            delete_user_data,
        )

        return Self(id, ctx)

    fn detach(self):
        _c.Diagnostics.mlirContextDetachDiagnosticHandler(self.ctx.c, self.id)

    @staticmethod
    fn c_handler(
        diagnostic: Diagnostic.cType, user_data: OpaquePointer
    ) -> _c.Support.MlirLogicalResult:
        var result = handler(diagnostic)
        return MlirLogicalResult(1 if result else 0)


@fieldwise_init
struct DiagnosticHandlerWithData[
    UserDataType: AnyType,
    handler: fn (Diagnostic, mut UserDataType) -> Bool,
    delete_user_data: fn (UnsafePointer[UserDataType]) -> None,
](Copyable, Movable):
    """Deals with attaching and detaching diagnostic functions along with user data to an MLIRContext.

    Parameters:
        UserDataType: The type of data being stored for use in the handler.
        handler: A function that handles a given Diagnostic with supporting user data.
        delete_user_data: A function that cleans up the stored user data if necessary.
    """

    var id: DiagnosticHandlerID
    var ctx: Context

    @staticmethod
    @always_inline
    fn attach(ctx: Context, init_data: UnsafePointer[UserDataType]) -> Self:
        var id = _c.Diagnostics.mlirContextAttachDiagnosticHandler(
            ctx.c,
            Self.c_handler,
            init_data.bitcast[NoneType](),
            Self.c_delete_user_data,
        )

        return Self(id, ctx)

    fn detach(self):
        _c.Diagnostics.mlirContextDetachDiagnosticHandler(self.ctx.c, self.id)

    @staticmethod
    fn c_delete_user_data(user_data: OpaquePointer):
        delete_user_data(user_data.bitcast[UserDataType]())

    @staticmethod
    fn c_handler(
        diagnostic: Diagnostic.cType, user_data: OpaquePointer
    ) -> _c.Support.MlirLogicalResult:
        var ptr = user_data.bitcast[UserDataType]()
        var result = handler(diagnostic, ptr[])
        return MlirLogicalResult(1 if result else 0)


struct ErrorCapturingDiagnosticHandler:
    """Captures the errors created via a DiagnosticHandler and raises them as mojo exceptions.
    """

    alias Handler = DiagnosticHandlerWithData[
        String, Self.set_error, Self.delete_user_data
    ]
    var ctx: Context
    var handler: Optional[Self.Handler]
    var error: String

    @implicit
    fn __init__(out self, ctx: Context):
        self.error = "MLIR raised but didn't set an error"
        self.ctx = ctx
        self.handler = None

    fn __enter__(mut self) raises:
        self.error = "MLIR raised but didn't set an error"
        if self.handler:
            raise "The same ErrorCapturingDiagnosticHandler instance cannot be entered multiple times at once."

        self.handler = Self.Handler.attach(
            self.ctx, UnsafePointer(to=self.error)
        )

    fn __exit__(mut self) raises:
        self.handler.unsafe_take().detach()
        self.handler = None

    fn __exit__(mut self, error: Error) raises -> Bool:
        self.handler.unsafe_take().detach()
        self.handler = None
        raise "MLIR Diagnostic: {}\nError: {}".format(self.error, error)

    @staticmethod
    fn set_error(diagnostic: Diagnostic, mut error: String) -> Bool:
        if diagnostic.get_severity() == DiagnosticSeverity.ERROR:
            error = String(diagnostic)
        return True

    @staticmethod
    # User data is dealt with by this classes lifetime. It does not need to be explicitly freed.
    fn delete_user_data(error_ptr: UnsafePointer[String]):
        pass
