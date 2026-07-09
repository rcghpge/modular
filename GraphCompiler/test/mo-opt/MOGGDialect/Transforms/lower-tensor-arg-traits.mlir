// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" 2>&1 | FileCheck %s --check-prefix=IMPORT
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" --mo-to-mogg --mogg-to-mgp 2>&1 | FileCheck %s

// === Baked tensor-arg contracts (distill matrix) ========================== //
//
// The first RUN line imports the trait_arg_trait_stubs package and asserts the
// per-argument `{role, fused, access, has_capability}` contract baked onto each
// kernel decl's `execute` KernelFunctionAttr as the trailing
// `tensor_arg_contracts` array. This is the import-time half of the trait-lane
// distill matrix; it folds in (and replaces) the former
// import-tensor-arg-traits.mlir, and together with the rejection cases in
// import-tensor-arg-traits-invalid.mlir replaces the deleted gtest
// `TestTensorArgContract.cpp`. Non-trait args (the DeviceContext and the
// implicit error/result refs) render as trailing `unit`.
//
// The fused-store / fused-transform / mutable-input-fused kernels distill
// cleanly at import even though dense-only lowering does not yet handle them,
// so they are asserted here (on the import run line) and intentionally do NOT
// appear as `mo.graph` lowering cases below.

// --- Role x dense (no fusion, no access form) --- //

// Input + DenseTensor -> input/none, and Output + DenseTensor -> output/none
// (the dense add covers both leading-output and trailing-input dense roles).
// IMPORT-DAG: registration = "trait_add_explicit"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "input"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// The `Some[...]` spelling distills to a byte-identical contract array (parity
// with the explicit-generic spelling above).
// IMPORT-DAG: registration = "trait_add_some"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "input"}, {access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// MutableInput + DenseTensor -> mutable_input/none (alongside a dense output).
// IMPORT-DAG: registration = "trait_mutable_input"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "none", fused = false, has_capability = false, role = "mutable_input"}, unit, unit, unit]

// Plain `Tensor` metadata (no `DenseTensor`): Input + Tensor -> input/none --
// the dense-vs-plain distinction is invisible to the distilled contract.
// IMPORT-DAG: registration = "trait_plain_tensor_input"{{.*}}[{access = "none", fused = false, has_capability = false, role = "input"}, unit, unit, unit]

// `TileTensorable` has no contract effect: Output + DenseTensor + TileTensorable
// -> output/none (identical to a plain dense output).
// IMPORT-DAG: registration = "trait_tile_tensorable_output"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, unit, unit, unit]

// --- Fused load (marker vs capability) --- //

// Input + Fused + LoadAccess (bare MARKER) -> input/load/fused, capability=false.
// IMPORT-DAG: registration = "trait_fused_load_marker"{{.*}}[{access = "load", fused = true, has_capability = false, role = "input"}, unit, unit, unit]

// Input + Fused + Loadable (CAPABILITY leaf) -> input/load/fused, capability=true.
// IMPORT-DAG: registration = "trait_fused_load"{{.*}}[{access = "none", fused = false, has_capability = false, role = "output"}, {access = "load", fused = true, has_capability = true, role = "input"}, unit, unit, unit]

// --- Fused store (marker vs capability) --- //

// Output + Fused + StoreAccess (bare MARKER) -> output/store/fused, cap=false.
// IMPORT-DAG: registration = "trait_fused_store_marker"{{.*}}[{access = "store", fused = true, has_capability = false, role = "output"}, unit, unit, unit]

// Output + Fused + Storable (CAPABILITY leaf) -> output/store/fused, cap=true.
// IMPORT-DAG: registration = "trait_fused_store_capability"{{.*}}[{access = "store", fused = true, has_capability = true, role = "output"}, unit, unit, unit]

// --- Fused transform (marker vs capability) --- //

// Output + Fused + TransformAccess (bare MARKER) -> output/transform/fused,
// cap=false.
// IMPORT-DAG: registration = "trait_fused_transform_marker"{{.*}}[{access = "transform", fused = true, has_capability = false, role = "output"}, unit, unit, unit]

// Output + Fused + Transformable (CAPABILITY leaf) -> output/transform/fused,
// cap=true.
// IMPORT-DAG: registration = "trait_fused_transform_capability"{{.*}}[{access = "transform", fused = true, has_capability = true, role = "output"}, unit, unit, unit]

// --- MutableInput can both load and store (one access form at a time) --- //

// MutableInput + Fused + Storable -> mutable_input/store/fused, capability=true.
// IMPORT-DAG: registration = "trait_mutable_fused_store"{{.*}}[{access = "store", fused = true, has_capability = true, role = "mutable_input"}, unit, unit, unit]

// MutableInput + Fused + Loadable -> mutable_input/load/fused, capability=true.
// IMPORT-DAG: registration = "trait_mutable_fused_load"{{.*}}[{access = "load", fused = true, has_capability = true, role = "mutable_input"}, unit, unit, unit]

// --- Non-fused access form --- //

// Input + DenseTensor + Loadable -> input/load with capability=true but
// fused=false: the access form is independent of the `Fused` marker.
// IMPORT-DAG: registration = "trait_nonfused_load_capability"{{.*}}[{access = "load", fused = false, has_capability = true, role = "input"}, unit, unit, unit]

// === LOWERING (dense-only): mogg.tensor_arg.__init__ + the legacy lane ===== //

// Dense CPU emit (Slice 3 / Phase 4): a trait tensor argument emits an inline
// `_MoggDenseTensorArg[...]` value constructed from the distilled contract; the
// kernel's generic `T` is inferred from it. The helper struct defs are emitted
// once into the module (so `DenseTensorArg` lives only inside the emitted Mojo,
// not as a committed library type). `--mgp-dump-mojo` prints the emitted text.
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" --mo-to-mogg --mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --mogg-emit-mojo --mgp-dump-mojo -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMIT
//
// The helper definitions are emitted once into the module: the role enum and
// the dense struct conforming to DenseTensor + Input/Output + TileTensorable.
// EMIT-DAG: struct _MoggTensorIO
// EMIT-DAG: struct _MoggDenseTensorArg[
// EMIT-DAG: TileTensorable,
//
// Each trait tensor arg constructs `_MoggDenseTensorArg[dtype, type_of(layout),
// alignment, _MoggTensorIO.<role>, device_passable](<opaque ptr>, layout)`, with
// the role read off the distilled contract (output / input / mutable_input all
// exercised by the graphs below) and `device_passable = False` for these host
// graphs. The kernel's generic `T` is inferred from the value.
// EMIT-DAG: _MoggDenseTensorArg[DType.float32, type_of({{.*}}), {{[0-9]+}}, _MoggTensorIO.OUTPUT, False]({{.*}}.unsafe_ptr().bitcast[NoneType](),
// EMIT-DAG: _MoggDenseTensorArg[DType.float32, type_of({{.*}}), {{[0-9]+}}, _MoggTensorIO.INPUT, False]({{.*}}.unsafe_ptr().bitcast[NoneType](),
// EMIT-DAG: _MoggDenseTensorArg[DType.float32, type_of({{.*}}), {{[0-9]+}}, _MoggTensorIO.MUTABLE_INPUT, False]({{.*}}.unsafe_ptr().bitcast[NoneType](),
//
// The trait kernel's `execute` is called with the constructed value (legacy
// `mogg.tensor.__init__` still emits for the legacy lane in mixed kernels).
// EMIT-DAG: TraitAddSome.execute[target = "cpu"]

// Slice 2 (Phase 3): a trait kernel's tensor arguments lower to
// `mogg.tensor_arg.__init__` carrying the distilled `#mogg.tensor_arg`
// contract, while a legacy `ManagedTensorSlice` kernel still lowers to
// `mogg.tensor.__init__`. The op + its attribute must be observable in the
// dumped IR after MOGG->MGP lowering (the stage where `buildMojoIOTensorObject`
// runs) but BEFORE emit-mojo (the stub emit is Slice 3). The contracts are
// stamped onto the `mogg.kernel` op during MO->MOGG lowering, so they appear on
// both the kernel and the per-arg init ops.

// A pure trait kernel: a dense output and two dense inputs (`trait_add_some`).
// Each tensor arg builds a `mogg.tensor_arg.__init__` carrying the role-bearing
// contract; there is NO `mogg.tensor.__init__` for this kernel. (After
// `--mogg-to-mgp` each graph becomes a `mgp.model @<name>` op.)
//
// Pin the op COUNT: both inputs (x, y) AND the single output must each build a
// `mogg.tensor_arg.__init__` -- three ops total. A single unordered match on
// the `input` contract would be satisfied by just one of the two inputs, so we
// count the two input contracts explicitly.
// CHECK-LABEL: mgp.model @trait_dense_add
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, has_capability = false}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, has_capability = false}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, has_capability = false}>
// CHECK-NOT: mogg.tensor.__init__
mo.graph @trait_dense_add(%x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_add_some", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// A trait kernel with a `mutable_input` argument (`trait_mutable_input`): a
// dense output `z` and a mutable input `x`. A mutable input crosses as an
// `!mo.buffer` operand, so its contract carries the `mutable_input` role. Both
// the output and the mutable input build `mogg.tensor_arg.__init__`. This case
// also exercises index alignment with TWO DIFFERENT input/output contracts
// (output vs mutable_input), so a swapped contract array would be caught.
// CHECK-LABEL: mgp.model @trait_mutable
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, has_capability = false}>
// CHECK-DAG: #mogg.tensor_arg<{role = mutable_input, fused = false, access = none, has_capability = false}>
// CHECK-NOT: mogg.tensor.__init__
mo.graph @trait_mutable(%x: !mo.buffer<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.chain.create()
  %0, %outch = mo.custom {symbol = "trait_mutable_input", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %c) : (!mo.buffer<[4], f32>, !mo.chain) -> (!mo.tensor<[4], f32>, !mo.chain)
  mo.output %0 : !mo.tensor<[4], f32>
}

// A multi-output trait kernel (`trait_multi_output`): two dense trait outputs
// `a`, `b` followed by a dense trait input `x`. This exercises the DPS-output
// contract array stamping for MORE THAN ONE output: two `role = output`
// contracts must each lower, plus one `role = input`. A single output-order
// entry would not catch a >1-output mis-stamp.
// CHECK-LABEL: mgp.model @trait_multi
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, has_capability = false}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, has_capability = false}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, has_capability = false}>
// CHECK-NOT: mogg.tensor.__init__
mo.graph @trait_multi(%x: !mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>, !mo.tensor<[5], f32>) {
  %a, %b = mo.custom {symbol = "trait_multi_output", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[5], f32>) -> (!mo.tensor<[5], f32>, !mo.tensor<[5], f32>)
  mo.output %a, %b : !mo.tensor<[5], f32>, !mo.tensor<[5], f32>
}

// A mixed legacy+trait kernel (`trait_mixed_legacy`): a trait dense output
// alongside two legacy `ManagedTensorSlice` inputs. The trait output lowers to
// `mogg.tensor_arg.__init__` (with the `output` contract); the legacy inputs
// lower to `mogg.tensor.__init__`. Both lanes coexist in the same kernel.
// CHECK-LABEL: mgp.model @mixed_legacy_trait
// CHECK-DAG: mogg.tensor_arg.__init__
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, has_capability = false}>
// CHECK-DAG: mogg.tensor.__init__
mo.graph @mixed_legacy_trait(%x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_mixed_legacy", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// Reverse mixed lane (`trait_mixed_legacy_out`): a LEGACY `OutputTensor`
// alongside two TRAIT `Input & DenseTensor` inputs. The legacy output lowers to
// `mogg.tensor.__init__`; the trait inputs lower to `mogg.tensor_arg.__init__`
// with `role = input`. Covers the legacy-output / trait-input coexistence
// direction (the mirror of `mixed_legacy_trait`).
// CHECK-LABEL: mgp.model @mixed_legacy_out
// CHECK-DAG: mogg.tensor.__init__
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, has_capability = false}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, has_capability = false}>
// CHECK-NOT: #mogg.tensor_arg<{role = output
mo.graph @mixed_legacy_out(%x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_mixed_legacy_out", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// Mixed lane with a legacy MUTABLE arg (`trait_mixed_legacy_mut`): a trait dense
// output `z` alongside a legacy `MutableInputTensor` `x` (a buffer operand). The
// legacy mutable input must still lower to `mogg.tensor.__init__` (the legacy
// `mut=true` lane is unchanged when a trait arg is present); the trait output
// lowers to `mogg.tensor_arg.__init__` with `role = output`.
// CHECK-LABEL: mgp.model @mixed_legacy_mut
// CHECK-DAG: mogg.tensor_arg.__init__
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, has_capability = false}>
// CHECK-DAG: mogg.tensor.__init__
mo.graph @mixed_legacy_mut(%x: !mo.buffer<[6], f32>) -> !mo.tensor<[6], f32> {
  %c = mo.chain.create()
  %0, %outch = mo.custom {symbol = "trait_mixed_legacy_mut", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %c) : (!mo.buffer<[6], f32>, !mo.chain) -> (!mo.tensor<[6], f32>, !mo.chain)
  mo.output %0 : !mo.tensor<[6], f32>
}

// Fused-load trait arg (`trait_mixed_legacy_fused`): a trait dense output, a
// legacy `InputTensor` (`mogg.tensor.__init__`), and a TRAIT fused/loadable
// input. The fused trait input's contract must FAITHFULLY carry `fused = true`,
// `access = load`, `has_capability = true` onto the op -- distinguishing it from
// the dense `fused = false, access = none` cases and catching any dropped/
// swapped boolean or mis-mapped access enum in `tensorArgContractToAttr`.
// CHECK-LABEL: mgp.model @mixed_legacy_fused
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, has_capability = false}>
// CHECK-DAG: #mogg.tensor_arg<{role = input, fused = true, access = load, has_capability = true}>
// CHECK-DAG: mogg.tensor.__init__
mo.graph @mixed_legacy_fused(%x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_mixed_legacy_fused", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// A pure-legacy control kernel (`legacy_only_control`): a legacy `OutputTensor`
// `z`, a legacy `InputTensor` `x`, and a legacy `MutableInputTensor` `w` (a
// buffer operand) -- NO trait args at all. It must lower to
// `mogg.tensor.__init__` only, with NO `mogg.tensor_arg.__init__` and NO
// `#mogg.tensor_arg` attribute anywhere -- the legacy lane is byte-for-byte
// unchanged.
// CHECK-LABEL: mgp.model @legacy_only
// CHECK: mogg.tensor.__init__
// CHECK-NOT: mogg.tensor_arg.__init__
// CHECK-NOT: #mogg.tensor_arg
mo.graph @legacy_only(%x: !mo.tensor<[8], f32>, %w: !mo.buffer<[8], f32>) -> !mo.tensor<[8], f32> {
  %c = mo.chain.create()
  %0, %outch = mo.custom {symbol = "legacy_only_control", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %w, %c) : (!mo.tensor<[8], f32>, !mo.buffer<[8], f32>, !mo.chain) -> (!mo.tensor<[8], f32>, !mo.chain)
  mo.output %0 : !mo.tensor<[8], f32>
}
