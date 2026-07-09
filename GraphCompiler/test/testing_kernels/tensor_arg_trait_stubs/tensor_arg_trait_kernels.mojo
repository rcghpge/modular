# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# Trait-bound kernel stubs for testing the tensor-argument trait import path
# (distilling a written trait composition into a `{role, fused, access,
# has_capability}` contract). Kept separate from MOGGKernelAPITest, which feeds
# the full emit/lowering/execution paths -- these stubs only exercise import and
# have no-op bodies, so they must not be mixed into the general kernel package.
#
from std.collections.string import StaticString
from std.gpu.host import DeviceContext

import extensibility as compiler
from extensibility import DenseTensor, Fused, Input, Loadable, Output


# Dense add, written with explicit trait-bound generic type parameters.
@compiler.register("trait_add_explicit")
struct TraitAddExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & DenseTensor,
        X: Input & DenseTensor,
    ](z: Z, x: X, y: X, ctx: DeviceContext) raises:
        pass


# Same contract as `TraitAddExplicit`, written with the `Some[...]` infer-only
# sugar. Must distill to an identical contract per argument.
@compiler.register("trait_add_some")
struct TraitAddSome:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[Input & DenseTensor],
        y: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass


# A fused, loadable input alongside a dense output -- exercises the fused-load
# access form (`Input & Fused & Loadable`).
@compiler.register("trait_fused_load")
struct TraitFusedLoad:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[Input & Fused & Loadable],
        ctx: DeviceContext,
    ) raises:
        pass


# Negative fixture: a trait-bound tensor arg passed with a non-borrowed (`mut`)
# convention. A tensor arg's mutability belongs in its contract (the role), not
# the calling convention, so VerifyKernels must reject this. It imports fine;
# only `mogg-verify-kernels` flags it.
@compiler.register("trait_bad_convention")
struct TraitBadConvention:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & DenseTensor,
    ](mut z: Z, ctx: DeviceContext) raises:
        pass


# Negative fixture: a trait-bound arg on `update_input_view`. Trait args are
# `execute`-only for now, so VerifyKernels rejects this even though the
# composition is well-formed and imports fine. (The `execute` here is a valid
# trait kernel; only `update_input_view` is rejected.)
@compiler.register("trait_update_view")
@compiler.view_kernel
struct TraitUpdateView:
    @staticmethod
    def update_input_view(input: Some[Input & DenseTensor]) -> Int:
        return 0

    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass
