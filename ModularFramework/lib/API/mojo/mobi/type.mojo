# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from .mobicc import Builder, TypePtr


@value
struct Type:
    """Wrapper over an MLIR type.

    This type has value semantics and can be copied around.
    Its implementation pointer field `t` is borrowed here and owned by the
    `Module` that created it.
    """

    # Pointer to `Type` implementation.
    var t: TypePtr
    var builder: Builder

    def __del__(owned self) -> None:
        self.builder.del_type(self.t)
