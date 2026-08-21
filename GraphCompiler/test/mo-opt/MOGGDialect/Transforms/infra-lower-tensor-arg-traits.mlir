// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg" 2>&1 | FileCheck %s --check-prefix=IMPORT
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg --mogg-to-mgp 2>&1 | FileCheck %s

// === Baked tensor-arg contracts (distill matrix) ========================== //
//
// The first RUN line imports the trait_arg_trait_stubs package and asserts the
// per-argument `{role, fused, access, capabilities}` contract baked onto each
// kernel decl's `execute` KernelFunctionAttr as the trailing
// `tensor_arg_contracts` array. `access`/`capabilities` are integer bitmasks
// of access forms (load = 1, store = 2, transform = 4; the empty set is 0).
// This is the import-time half of the trait-lane
// distill matrix; it folds in (and replaces) the former
// import-tensor-arg-traits.mlir, and together with the rejection cases in
// infra-import-tensor-arg-traits-invalid.mlir replaces the deleted gtest
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
// IMPORT-DAG: registration = "trait_add_explicit"{{.*}}[{access = 0 : i32, capabilities = 0 : i32, fused = false, role = "output"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}, unit, unit, unit]

// The `Some[...]` spelling distills to a byte-identical contract array (parity
// with the explicit-generic spelling above).
// IMPORT-DAG: registration = "trait_add_some"{{.*}}[{access = 0 : i32, capabilities = 0 : i32, fused = false, role = "output"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}, unit, unit, unit]

// MutableInput + DenseTensor -> mutable_input/none (alongside a dense output).
// IMPORT-DAG: registration = "trait_mutable_input"{{.*}}[{access = 0 : i32, capabilities = 0 : i32, fused = false, role = "output"}, {access = 0 : i32, capabilities = 0 : i32, fused = false, role = "mutable_input"}, unit, unit, unit]

// Plain `Tensor` metadata (no `DenseTensor`): Input + Tensor -> input/none --
// the dense-vs-plain distinction is invisible to the distilled contract.
// IMPORT-DAG: registration = "trait_plain_tensor_input"{{.*}}[{access = 0 : i32, capabilities = 0 : i32, fused = false, role = "input"}, unit, unit, unit]

// `TileTensorable` has no contract effect: Output + DenseTensor + TileTensorable
// -> output/none (identical to a plain dense output).
// IMPORT-DAG: registration = "trait_tile_tensorable_output"{{.*}}[{access = 0 : i32, capabilities = 0 : i32, fused = false, role = "output"}, unit, unit, unit]

// --- Fused load (marker vs capability) --- //

// Input + Fused + LoadAccess (bare MARKER) -> input/load/fused, capability=0.
// IMPORT-DAG: registration = "trait_fused_load_marker"{{.*}}[{access = 1 : i32, capabilities = 0 : i32, fused = true, role = "input"}, unit, unit, unit]

// Input + Fused + Loadable (CAPABILITY leaf) -> input/load/fused, capability=1.
// IMPORT-DAG: registration = "trait_fused_load"{{.*}}[{access = 0 : i32, capabilities = 0 : i32, fused = false, role = "output"}, {access = 1 : i32, capabilities = 1 : i32, fused = true, role = "input"}, unit, unit, unit]

// --- Fused store (marker vs capability) --- //

// Output + Fused + StoreAccess (bare MARKER) -> output/store/fused, cap=0.
// IMPORT-DAG: registration = "trait_fused_store_marker"{{.*}}[{access = 2 : i32, capabilities = 0 : i32, fused = true, role = "output"}, unit, unit, unit]

// Output + Fused + Storable (CAPABILITY leaf) -> output/store/fused, cap=store.
// IMPORT-DAG: registration = "trait_fused_store_capability"{{.*}}[{access = 2 : i32, capabilities = 2 : i32, fused = true, role = "output"}, unit, unit, unit]

// --- Fused transform (marker vs capability) --- //

// Output + Fused + TransformAccess (bare MARKER) -> output/transform/fused,
// cap=0.
// IMPORT-DAG: registration = "trait_fused_transform_marker"{{.*}}[{access = 4 : i32, capabilities = 0 : i32, fused = true, role = "output"}, unit, unit, unit]

// Output + Fused + Transformable (CAPABILITY leaf) -> output/transform/fused,
// cap=transform.
// IMPORT-DAG: registration = "trait_fused_transform_capability"{{.*}}[{access = 4 : i32, capabilities = 4 : i32, fused = true, role = "output"}, unit, unit, unit]

// --- MutableInput can both load and store (one access form at a time) --- //

// MutableInput + Fused + Storable -> mutable_input/store/fused, capability=store.
// IMPORT-DAG: registration = "trait_mutable_fused_store"{{.*}}[{access = 2 : i32, capabilities = 2 : i32, fused = true, role = "mutable_input"}, unit, unit, unit]

// MutableInput + Fused + Loadable -> mutable_input/load/fused, capability=load.
// IMPORT-DAG: registration = "trait_mutable_fused_load"{{.*}}[{access = 1 : i32, capabilities = 1 : i32, fused = true, role = "mutable_input"}, unit, unit, unit]

// --- Non-fused access form --- //

// Input + DenseTensor + Loadable -> input/load with capability=load but
// fused=false: the access form is independent of the `Fused` marker.
// IMPORT-DAG: registration = "trait_nonfused_load_capability"{{.*}}[{access = 1 : i32, capabilities = 1 : i32, fused = false, role = "input"}, unit, unit, unit]

// MutableInput + DenseTensor + Loadable + Storable -> mutable_input carrying BOTH
// access forms at once: access = 3 (load|store) and capabilities = 3. The
// positive proof that a contract holds the full access SET (multiple forms on
// one argument); a swapped/dropped form or a single-form fallback would be
// caught here. Emitting the dense element-access conformances is a later slice.
// IMPORT-DAG: registration = "trait_nonfused_load_store_capability"{{.*}}[{access = 3 : i32, capabilities = 3 : i32, fused = false, role = "mutable_input"}, unit, unit, unit]

// --- Shape-function no-role flavor --- //

// `Some[Tensor]` (`trait_tensor_shape_fn`) -> bound = "tensor".
// IMPORT-DAG: mogg.shape_func.decl{{.*}}registration = "trait_tensor_shape_fn"{{.*}}[{access = 0 : i32, bound = "tensor", capabilities = 0 : i32, fused = false, role = "none"}]

// `Some[DenseTensor]` (`trait_metadata_shape_fn`) -> bound = "dense".
// IMPORT-DAG: mogg.shape_func.decl{{.*}}registration = "trait_metadata_shape_fn"{{.*}}[{access = 0 : i32, bound = "dense", capabilities = 0 : i32, fused = false, role = "none"}]

// `Some[TileTensorable]` (`trait_tile_tensorable_shape_fn`) -> bound = "tile".
// IMPORT-DAG: mogg.shape_func.decl{{.*}}registration = "trait_tile_tensorable_shape_fn"{{.*}}[{access = 0 : i32, bound = "tile", capabilities = 0 : i32, fused = false, role = "none"}]

// === LOWERING (dense-only): mogg.tensor_arg.__init__ + the legacy lane ===== //

// Dense CPU emit: region→stub passes `_MoggTensorBufferCapture`; the stub
// assembles `_MoggDenseTensorArg` before forwarding to `execute`.
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg --mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --mogg-emit-mojo --mgp-dump-mojo -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMIT
//
// Buffer capture at the region boundary and dense assembly inside the stub.
// EMIT-DAG: struct _MoggTensorBufferCapture[
// EMIT-DAG: struct _MoggTensorIO
// EMIT-DAG: struct _MoggDenseTensorArg[
// EMIT-DAG: TileTensorable where not _is_device_passable,
// EMIT-DAG: ] where not Self._is_device_passable:
//
// Region constructs buffer captures (not execute-facing dense args).
// EMIT-DAG: _MoggTensorBufferCapture[DType.float32, type_of(
// EMIT-DAG: unsafe_origin_cast[MutUntrackedOrigin]()
//
// Dynamic captures infer layout types from arguments because runtime extents
// are unavailable in the stub signature.
// EMIT-DAG: __trait_capture_layout_type_{{[0-9]+}} : layout.TensorLayout
// EMIT-DAG: _MoggTensorBufferCapture[DType.float32, __trait_capture_layout_type_
//
// Keep each capture paired with its corresponding layout parameter.
// EMIT-DAG: __trait_capture_layout_type_2 : layout.TensorLayout](arg__0: _MoggTensorBufferCapture[DType.float32, __trait_capture_layout_type_0], arg__1: _MoggTensorBufferCapture[DType.float32, __trait_capture_layout_type_1], arg__2: _MoggTensorBufferCapture[DType.float32, __trait_capture_layout_type_2]
//
// EMIT-DAG: layout.Coord(layout.Idx[2], Scalar[DType.int](
//
// Stub assembles dense execute args from capture before `execute`.
// EMIT-DAG: _MoggDenseTensorArg[type_of(arg__
// EMIT-DAG: _MoggTensorIO.OUTPUT
// EMIT-DAG: _MoggTensorIO.INPUT
// EMIT-DAG: _MoggTensorIO.MUTABLE_INPUT
//
// The trait kernel's `execute` is called with the assembled dense values.
// EMIT-DAG: TraitAddSome.execute[target = "cpu"]
//
// Fused-store trait kernel (full pipeline): stub builds fusion wrapper from capture.
// EMIT-DAG: _MoggOutputStoreFusionTensor[
// EMIT-DAG: TraitFusedStoreDouble.execute[target = "cpu"]

// A trait kernel's tensor arguments lower to `mogg.tensor_arg.__init__`
// carrying the distilled `#mogg.tensor_arg` contract, while a legacy
// `ManagedTensorSlice` kernel still lowers to `mogg.tensor.__init__`. The op
// + its attribute must be observable in the dumped IR after MOGG->MGP
// lowering (the stage where `buildMojoIOTensorObject` runs) but before
// emit-mojo. The contract is derived from the kernel declaration at boundary
// lowering (GEX-4071), so it appears only on the per-arg init ops.

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
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// CHECK-NOT: mogg.tensor.__init__
mo.graph @trait_dense_add(%x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_add_some", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// A trait kernel with a `mutable_input` argument (`trait_mutable_input`): a
// dense output `z` and a mutable input `x`. A mutable input crosses as an
// `!mo.buffer` operand, so its contract carries the `mutable_input` role. Both
// the output and the mutable input build `mogg.tensor_arg.__init__`. This case
// also exercises per-arg derivation with TWO DIFFERENT input/output contracts
// (output vs mutable_input), so a swapped operand-position lookup would be
// caught.
// CHECK-LABEL: mgp.model @trait_mutable
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: #mogg.tensor_arg<{role = mutable_input, fused = false, access = none, capabilities = none}>
// CHECK-NOT: mogg.tensor.__init__
mo.graph @trait_mutable(%x: !mo.buffer<[4], f32>) -> !mo.tensor<[4], f32> {
  %c = mo.chain.create()
  %0, %outch = mo.custom {symbol = "trait_mutable_input", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %c) : (!mo.buffer<[4], f32>, !mo.chain) -> (!mo.tensor<[4], f32>, !mo.chain)
  mo.output %0 : !mo.tensor<[4], f32>
}

// A multi-output trait kernel (`trait_multi_output`): two dense trait outputs
// `a`, `b` followed by a dense trait input `x`. This exercises DPS-output
// contract derivation for MORE THAN ONE output: two `role = output` contracts
// must each lower, plus one `role = input`. A single output would not catch a
// >1-output position mixup.
// CHECK-LABEL: mgp.model @trait_multi
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
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
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
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
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
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
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor.__init__
mo.graph @mixed_legacy_mut(%x: !mo.buffer<[6], f32>) -> !mo.tensor<[6], f32> {
  %c = mo.chain.create()
  %0, %outch = mo.custom {symbol = "trait_mixed_legacy_mut", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %c) : (!mo.buffer<[6], f32>, !mo.chain) -> (!mo.tensor<[6], f32>, !mo.chain)
  mo.output %0 : !mo.tensor<[6], f32>
}

// Fused-load trait arg (`trait_mixed_legacy_fused`): a trait dense output, a
// legacy `InputTensor` (`mogg.tensor.__init__`), and a TRAIT fused/loadable
// input. The fused trait input's contract must FAITHFULLY carry `fused = true`,
// `access = load`, `capabilities = load` onto the op -- distinguishing it from
// the dense `fused = false, access = none` cases and catching any dropped/
// swapped boolean or mis-mapped access enum in `tensorArgContractToAttr`.
// CHECK-LABEL: mgp.model @mixed_legacy_fused
// CHECK-DAG: #mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: #mogg.tensor_arg<{role = input, fused = true, access = load, capabilities = load}>
// CHECK-DAG: mogg.tensor.__init__
mo.graph @mixed_legacy_fused(%x: !mo.tensor<[2, 3], f32>, %y: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %0 = mo.custom {symbol = "trait_mixed_legacy_fused", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// Fused-store trait arg (`trait_fused_store_double`): validated on the EMIT run
// (full gc pipeline) below.
mo.graph @trait_fused_store(%x: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %0 = mo.custom {symbol = "trait_fused_store_double", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x) : (!mo.tensor<[4], f32>) -> !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// Mixed static/dynamic shapes exercise layout inference for three captures.
// CHECK-LABEL: mgp.model @trait_dense_dyn
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// CHECK-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
mo.graph @trait_dense_dyn<D0>(%x: !mo.tensor<[2, D0], f32>, %y: !mo.tensor<[2, D0], f32>) -> !mo.tensor<[2, D0], f32> {
  %0 = mo.custom {symbol = "trait_add_some", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %y) : (!mo.tensor<[2, D0], f32>, !mo.tensor<[2, D0], f32>) -> !mo.tensor<[2, D0], f32>
  mo.output %0 : !mo.tensor<[2, D0], f32>
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
