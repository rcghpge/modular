// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" --mogg-dump-kernels 2>&1 | FileCheck %s

// Downstream role plumbing for trait-bound tensor arguments. After import bakes
// the per-argument contract, `KernelDeclAdaptor` (via `countDPSOutputs` /
// `classifyArgType`) must read the contract `role` to split the signature into
// DPS outputs (`[out]`) vs inputs (`[in]`). Before the fix, a trait arg was
// non-struct -> classified `Unsupported` -> the leading-output count broke at
// the first trait arg, so a trait `Output` was mis-mapped into the inputs.
//
// `mogg-dump-kernels` prints the decoded adaptor, so the `[out]`/`[in]` split
// and the per-arg `mut` are directly observable. Kernels are dumped in
// alphabetical registration order, so the CHECK-LABELs below follow that order.

// Pure-legacy control (NO trait args): a regression guard that this change
// leaves the legacy `[out]`/`[in]`/`mut` split byte-for-byte unchanged. A
// legacy `OutputTensor` is the DPS output, a read-only `InputTensor` is
// `[in] mut=false`, and a legacy `MutableInputTensor` is `[in] mut=true`.
// CHECK-LABEL: @register(legacy_only_control)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  w: Tensor[mut=true, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Dense add (explicit `[T: Output & DenseTensor](z: T)` spelling): the trait
// output is classified as a DPS output, the two trait inputs as borrowed
// (`mut=false`) inputs -- the output is NOT folded into the inputs.
// CHECK-LABEL: @register(trait_add_explicit)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  y: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// The `Some[...]` sugar classifies identically to the explicit-generic spelling
// above (D6 parity): same `[out]`/`[in]` split, same per-arg facts.
// CHECK-LABEL: @register(trait_add_some)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  y: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// All-outputs: three leading trait outputs and nothing else, so the leading-
// output count runs to `n` without ever breaking -- covers count > 2 and the
// run-to-end path.
// CHECK-LABEL: @register(trait_all_outputs)
// CHECK:       execute  (
// CHECK-NEXT:    [out] a: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [out] b: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [out] c: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// A fused, loadable input alongside a dense output: still split correctly.
// NOTE: the imported contract for `x` is `{fused=true, access=load}` (see
// import-tensor-arg-traits.mlir), but the adaptor reports `fused=false` here.
// This `fused=true -> fused=false` collapse is the INTENTIONAL dense-only
// behavior of this slice: `traitContractToTensorAdaptor` maps `output ->
// OutputTensor` / `input -> InputTensor` and does NOT yet propagate the
// contract's `fused`/`access` bits into the adaptor (deferred to the fusion
// slice). When fused propagation lands, this `fused=false` MUST be consciously
// updated to `fused=true`; it is not a genuine non-fused arg.
// CHECK-LABEL: @register(trait_fused_load)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// D6 parity for the fused-load role: the explicit-generic spelling
// (`[X: Input & Fused & Loadable]`) classifies identically to the `Some[...]`
// twin above (same dense-only `fused=false` collapse).
// CHECK-LABEL: @register(trait_fused_load_explicit)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Leading `mutable_input`: the FIRST tensor arg is writable but not an output,
// so `countDPSOutputs` breaks at position 0 and yields ZERO DPS outputs -- both
// tensors are `[in]`, and the leading mutable carries `mut=true`.
// CHECK-LABEL: @register(trait_leading_mutable)
// CHECK:       execute  (
// CHECK-NEXT:    [in]  x: Tensor[mut=true, fused=false],
// CHECK-NEXT:    [in]  y: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Mixed lane: a trait output coexists with legacy `ManagedTensorSlice`
// (`InputTensor`) inputs. Both lanes classify independently; the legacy lane is
// unchanged.
// CHECK-LABEL: @register(trait_mixed_legacy)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  y: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Mixed lane with a FUSED trait arg: a legacy `InputTensor` coexists with a
// trait `Input & Fused & Loadable`, confirming the two lanes coexist when a
// fused trait arg (not just a dense one) is present.
// CHECK-LABEL: @register(trait_mixed_legacy_fused)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  y: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Mixed lane with a legacy MUTABLE arg: a legacy `MutableInputTensor`
// (`mut=true`) coexists with a trait output. The trait output is `[out]`, the
// legacy mutable arg still renders `[in] mut=true`.
// CHECK-LABEL: @register(trait_mixed_legacy_mut)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=true, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Mixed lane, other direction: a legacy `OutputTensor` is the DPS output and
// trait inputs are borrowed inputs.
// CHECK-LABEL: @register(trait_mixed_legacy_out)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  y: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Multi-output: two leading trait outputs are BOTH counted as DPS outputs; the
// trailing trait input is an input. (This is the case the old `break-on-first-
// non-struct` semantics could not express.)
// CHECK-LABEL: @register(trait_multi_output)
// CHECK:       execute  (
// CHECK-NEXT:    [out] a: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [out] b: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// D6 parity for the multi-output role: the explicit-generic spelling classifies
// identically to the `Some[...]` twin above (two leading `[out]`, then `[in]`).
// CHECK-LABEL: @register(trait_multi_output_explicit)
// CHECK:       execute  (
// CHECK-NEXT:    [out] a: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [out] b: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// `MutableInput` role: a writable, non-DPS-output tensor. It is an input
// (`[in]`) but carries `mut=true`, distinct from a read-only `Input`
// (`mut=false`) -- matching the legacy `MutableInputTensor` IOSpec.
// CHECK-LABEL: @register(trait_mutable_input)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=true, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// D6 parity for the mutable-input role: the explicit-generic spelling
// classifies identically to the `Some[...]` twin above (`[in] mut=true`).
// CHECK-LABEL: @register(trait_mutable_input_explicit)
// CHECK:       execute  (
// CHECK-NEXT:    [out] z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  x: Tensor[mut=true, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

// Non-leading output: input, output, input. `countDPSOutputs` breaks at the
// FIRST input (position 0), so the later output is NOT counted as a DPS output
// (it renders `[in]`) -- mirroring the legacy struct lane's "break at first
// non-output" rule.
// CHECK-LABEL: @register(trait_non_leading_output)
// CHECK:       execute  (
// CHECK-NEXT:    [in]  x: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  z: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  y: Tensor[mut=false, fused=false],
// CHECK-NEXT:    [in]  ctx: Context,

module {}
