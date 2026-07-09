# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Trait-bound kernel stubs whose signatures are INVALID -- each violates exactly
# one tensor-argument validation rule. They confirm that the import-time
# validation rejects every bad composition (and reports them all, rather than
# aborting on the first). Kept in a separate package from the valid stubs
# because an invalid composition fails the package import.
#
from std.collections.string import StaticString
from std.gpu.host import DeviceContext
from std.utils import IndexList

from layout import coord_to_index_list

import extensibility as compiler
from extensibility import (
    DenseTensor,
    Fused,
    Input,
    Output,
    Storable,
)


# No role trait (none of Input / Output / MutableInput).
@compiler.register("trait_no_role")
struct TraitNoRole:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[DenseTensor], ctx: DeviceContext) raises:
        pass


# More than one role trait.
@compiler.register("trait_two_roles")
struct TraitTwoRoles:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Input & Output & DenseTensor,
    ](z: Z, ctx: DeviceContext) raises:
        pass


# Fused but with no access form.
@compiler.register("trait_fused_no_access")
struct TraitFusedNoAccess:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Output & Fused & DenseTensor], ctx: DeviceContext) raises:
        pass


# A foreign (non-tensor-argument) trait in the bound -- `Copyable` is a stdlib
# trait, not part of extensibility.tensor_arg_traits.
@compiler.register("trait_foreign")
struct TraitForeign:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & DenseTensor & Copyable,
    ](z: Z, ctx: DeviceContext) raises:
        pass


# Store access on a read-only Input.
@compiler.register("trait_store_on_input")
struct TraitStoreOnInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](z: Some[Input & Fused & Storable], ctx: DeviceContext) raises:
        pass


# A trait-bound argument on a shape function. Trait args are `execute`-only;
# shape functions are rejected at import (they are a separate decl op, seen in
# the import shape-function loop regardless of whether a kernel is registered).
@compiler.register_shape_function("trait_shape_fn")
def trait_shape_fn[T: Input & DenseTensor](x: T) -> IndexList[T.rank]:
    return rebind[IndexList[T.rank]](coord_to_index_list(x.shape().tuple()))
