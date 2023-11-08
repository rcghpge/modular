# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug import trap
from .mobicc import TypePtr


@value
@register_passable
struct TensorType:
    """Wrapper over an MLIR type. It represents a mo Tensor type variant.

    This type has value semantics and can be copied around.
    Its implementation pointer field `t` is borrowed here and owned by the
    `Module` that created it.
    """

    var t: TypePtr


@value
struct Type:
    """Wrapper over an MLIR type. It represents a mo Type object.

    This type has value semantics and can be copied around.
    Its implementation pointer field `t` is borrowed here and owned by the
    `Module` that created it.

    MO types are commonly tensors but can be other things!
    """

    alias tensor: Int = 0
    alias type = __mlir_type[
        `!kgen.variant<`,
        TensorType,
        `, `,
        `i1`,  # placeholder for other types
        `>`,
    ]

    var variant: Self.type

    fn __init__(inout self, type: TypePtr):
        self.variant = __mlir_op.`kgen.variant.create`[
            _type = Self.type, index = Self.tensor.value
        ](TensorType(type))

    fn unwrap_tensor(self) -> TensorType:
        if not Bool(
            __mlir_op.`kgen.variant.is`[index = Self.tensor.value](self.variant)
        ):
            print("Tried to unwrap_tensor on non-tensor!")
            trap()
        return __mlir_op.`kgen.variant.get`[index = Self.tensor.value](
            self.variant
        )

    fn type_ptr(self) -> TypePtr:
        return self.unwrap_tensor().t
