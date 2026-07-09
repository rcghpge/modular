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
from extensibility import (
    DenseTensor,
    Fused,
    Input,
    InputTensor,
    Loadable,
    MutableInput,
    Output,
    OutputTensor,
    _MutableInputTensor as MutableInputTensor,
)


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


# Negative fixture (off-by-one guard for the trait exclusion in the legacy
# borrow-check): a signature with a trait arg at position 0 (borrowed, fine) AND
# a legacy `ManagedTensorSlice` (`OutputTensor`) at position 1 passed with a
# non-borrowed (`mut`) convention. The legacy borrow-check must STILL fire for
# the legacy arg at position 1; the `!getTensorArgTraitType(...)` exclusion must
# skip ONLY the trait arg at position 0. An off-by-one there would silently
# suppress the legacy diagnostic.
@compiler.register("trait_legacy_bad_convention")
struct TraitLegacyBadConvention:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[Input & DenseTensor],
        mut z: OutputTensor,
        ctx: DeviceContext,
    ) raises:
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


# Two dense trait outputs followed by a dense trait input -- exercises the
# multi-output DPS split: both leading outputs must be classified as DPS
# outputs (not folded into the inputs).
@compiler.register("trait_multi_output")
struct TraitMultiOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        a: Some[Output & DenseTensor],
        b: Some[Output & DenseTensor],
        x: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass


# A dense output and a `MutableInput` trait argument -- the mutable-input role
# must classify as a mutable (writable, non-DPS-output) tensor, distinct from a
# read-only `Input`.
@compiler.register("trait_mutable_input")
struct TraitMutableInput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: Some[MutableInput & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass


# Mixed lane: a trait-bound dense output alongside legacy `ManagedTensorSlice`
# (`InputTensor`) inputs. The trait output and the legacy inputs must classify
# independently and coexist -- the legacy lane is unchanged.
@compiler.register("trait_mixed_legacy")
struct TraitMixedLegacy:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: InputTensor,
        y: InputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# Mirror of `trait_mixed_legacy` with a legacy `OutputTensor` and trait inputs:
# the legacy output classifies as the DPS output, the trait inputs as borrowed
# inputs.
@compiler.register("trait_mixed_legacy_out")
struct TraitMixedLegacyOut:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: Some[Input & DenseTensor],
        y: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass


# D6 parity: explicit-generic spellings of the multi-output / mutable-input /
# fused-load roles. Each must classify identically to its `Some[...]` twin
# above, proving the explicit and inferred spellings agree for ALL roles, not
# just the dense add pair.
@compiler.register("trait_multi_output_explicit")
struct TraitMultiOutputExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        A: Output & DenseTensor,
        X: Input & DenseTensor,
    ](a: A, b: A, x: X, ctx: DeviceContext) raises:
        pass


@compiler.register("trait_mutable_input_explicit")
struct TraitMutableInputExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & DenseTensor,
        X: MutableInput & DenseTensor,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


@compiler.register("trait_fused_load_explicit")
struct TraitFusedLoadExplicit:
    @staticmethod
    def execute[
        target: StaticString,
        Z: Output & DenseTensor,
        X: Input & Fused & Loadable,
    ](z: Z, x: X, ctx: DeviceContext) raises:
        pass


# Leading mutable_input: the FIRST tensor arg is a (writable, non-output) input,
# so `countDPSOutputs` must break at position 0 and yield ZERO DPS outputs --
# directly exercising the "no leading output" boundary for trait args.
@compiler.register("trait_leading_mutable")
struct TraitLeadingMutable:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[MutableInput & DenseTensor],
        y: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass


# All-outputs: three leading trait outputs and nothing else, so the leading-
# output loop runs to `n` without ever breaking -- covers count > 2 and the
# run-to-end path.
@compiler.register("trait_all_outputs")
struct TraitAllOutputs:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        a: Some[Output & DenseTensor],
        b: Some[Output & DenseTensor],
        c: Some[Output & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass


# Non-leading output: input, output, input. `countDPSOutputs` must break at the
# FIRST input (position 0) and NOT count the later output as a DPS output --
# mirroring the legacy struct lane's "break at first non-output" rule.
@compiler.register("trait_non_leading_output")
struct TraitNonLeadingOutput:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        x: Some[Input & DenseTensor],
        z: Some[Output & DenseTensor],
        y: Some[Input & DenseTensor],
        ctx: DeviceContext,
    ) raises:
        pass


# Mixed lane with a legacy MUTABLE arg: a legacy `MutableInputTensor`
# (`mut=true`, non-DPS-output input) coexists with a trait output. The legacy
# mutable arg must still render `[in] mut=true` and the trait output `[out]`.
@compiler.register("trait_mixed_legacy_mut")
struct TraitMixedLegacyMut:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: MutableInputTensor,
        ctx: DeviceContext,
    ) raises:
        pass


# Mixed lane with a FUSED trait arg: a legacy `InputTensor` coexists with a
# trait `Input & Fused & Loadable`. Confirms the two lanes coexist when a fused
# trait arg (not just a dense one) is present.
@compiler.register("trait_mixed_legacy_fused")
struct TraitMixedLegacyFused:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: Some[Output & DenseTensor],
        x: InputTensor,
        y: Some[Input & Fused & Loadable],
        ctx: DeviceContext,
    ) raises:
        pass


# Pure-legacy control: no trait args at all. A regression guard that this change
# leaves the legacy `[out]`/`[in]`/`mut` split byte-for-byte unchanged. The
# `MutableInputTensor` here pins `mut=true` on the legacy lane too.
@compiler.register("legacy_only_control")
struct LegacyOnlyControl:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        z: OutputTensor,
        x: InputTensor,
        w: MutableInputTensor,
        ctx: DeviceContext,
    ) raises:
        pass
