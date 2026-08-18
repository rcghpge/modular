// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --cse --mo-to-mogg --mogg-simplify-kernel-operands --mogg-fuse-consts --canonicalize 2>&1 | FileCheck %s --check-prefix=FOLD
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --mo-to-mogg --mogg-fuse-consts --mogg-to-mgp 2>&1 | FileCheck %s --check-prefix=CONSTSTUB
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --cse --mo-to-mogg --mogg-simplify-kernel-operands --mogg-fuse-consts --canonicalize --mogg-to-mgp 2>&1 | FileCheck %s --check-prefix=CONSTMODEL
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --cse --mo-to-mogg --mogg-simplify-kernel-operands --mogg-fuse-consts --canonicalize --mogg-to-mgp --mogg-outline-kernels --mogg-add-try-except-blocks --mogg-emit-mojo --mgp-dump-mojo -o /dev/null 2>&1 | FileCheck %s --check-prefix=EMITCONST
// RUN: mo-opt %s --mogg-import-packages="mojo-pkg-paths=%tensor_arg_trait_pkg mojo-pkg-paths=%kernel_api_test_pkg" --cse --mo-to-mogg --mogg-fuse-elementwise --mogg-fuse-prologues --mogg-simplify-kernel-operands --mogg-fuse-consts --canonicalize --mogg-to-mgp 2>&1 | FileCheck %s --check-prefix=PROLOG

// Stub-side checks come before the model checks because every
// `mogg.kernel.stub` prints before the first `mgp.model`.
// `@trait_constant_shared_lanes`: one buffer, one init per lane.
// CONSTMODEL-LABEL: mogg.kernel.stub @custom__trait_mixed_legacy_fused_mogg (
// CONSTMODEL: mogg.tensor_arg.__init__(%[[SHARED_BUF:arg[0-9]+]]) {contract = #mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// CONSTMODEL: mogg.tensor.__init__(%[[SHARED_BUF]])

// A dense trait input folds to an in-stub capture and execute-facing dense arg.
// CONSTSTUB-LABEL: mogg.kernel.stub @custom__trait_add_explicit
// CONSTSTUB: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// EMITCONST: def custom__trait_add_explicit_mogg{{[_0-9]*}}[{{.*}}]({{.*}}MutByteBuffer) raises:
// EMITCONST: [[DCAP:var___[0-9]+]] = _MoggTensorBufferCapture[DType.float32, {{.*}}](
// EMITCONST: _MoggDenseTensorArg[type_of([[DCAP]]).dtype, {{.*}}_MoggTensorIO.INPUT, False]
// CONSTMODEL-LABEL: mgp.model @trait_constant_dense
// CONSTMODEL-DAG: mgp.buffer.constant
// CONSTMODEL-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// CONSTMODEL-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
mo.graph @trait_constant_dense(%x: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %cst = mo.constant {value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %0 = mo.custom {symbol = "trait_add_explicit", target = "cpu", device = #M.device_ref<"cpu", 0>}(%x, %cst) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// A fused trait input folds to the capture backing its fusion wrapper.
// CONSTSTUB-LABEL: mogg.kernel.stub @custom__trait_fused_load
// CONSTSTUB: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = false, access = none, capabilities = none}>
// EMITCONST: def custom__trait_fused_load_mogg{{[_0-9]*}}[{{.*}}]({{.*}}MutByteBuffer) raises:
// EMITCONST: [[CAP:var___[0-9]+]] = _MoggTensorBufferCapture[DType.float32, {{.*}}](
// EMITCONST: InputLoadFunctor_{{[0-9]+}}[type_of([[CAP]]).dtype
// EMITCONST: _MoggInputLoadFusionTensor[type_of([[CAP]]).dtype
// CONSTMODEL-LABEL: mgp.model @trait_constant_fused
// CONSTMODEL-DAG: mgp.buffer.constant
// CONSTMODEL-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
mo.graph @trait_constant_fused() -> !mo.tensor<[2, 3], f32> {
  %cst = mo.constant {value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %0 = mo.custom {symbol = "trait_fused_load", target = "cpu", device = #M.device_ref<"cpu", 0>}(%cst) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// Folding the leading legacy operand must not disturb the trailing trait
// operand's derived contract.
// CONSTMODEL-LABEL: mgp.model @trait_constant_legacy_fold
// CONSTMODEL-DAG: mgp.buffer.constant
// CONSTMODEL-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = input, fused = true, access = load, capabilities = load}>
// CONSTMODEL-DAG: mogg.tensor_arg.__init__{{.*}}#mogg.tensor_arg<{role = output, fused = false, access = none, capabilities = none}>
mo.graph @trait_constant_legacy_fold(%y: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %cst = mo.constant {value = #M.dense_array<1., 2., 3., 4., 5., 6.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %0 = mo.custom {symbol = "trait_mixed_legacy_fused", target = "cpu", device = #M.device_ref<"cpu", 0>}(%cst, %y) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// A constant operand brought in by prologue fusion must also fold: both
// captures materialize in the prologue stub, and both buffers reach the model.
// PROLOG-LABEL: mogg.kernel.stub @Prologue_add_custom__trait_fuse
// PROLOG: mogg.tensor_arg.__init__(%arg
// PROLOG-NEXT: mogg.tensor_arg.__init__(%arg
// PROLOG-LABEL: mgp.model @trait_constant_producer
// PROLOG-COUNT-2: mgp.buffer.constant
mo.graph @trait_constant_producer() -> !mo.tensor<[2, 3], f32> {
  %lhs = mo.constant {value = #M.dense_array<7., 7., 7., 7., 7., 7.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %rhs = mo.constant {value = #M.dense_array<8., 8., 8., 8., 8., 8.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %p = mo.add(%lhs, %rhs) : !mo.tensor<[2, 3], f32>
  %0 = mo.custom {symbol = "trait_fused_load", target = "cpu", device = #M.device_ref<"cpu", 0>}(%p) : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CSE may merge equal legacy and trait constants onto one in-stub value.
// FOLD-LABEL: mo.graph @trait_constant_shared_lanes
// FOLD: mogg.kernel()
// FOLD-COUNT-1: mo.constant
// FOLD-NOT: mo.constant
// CONSTMODEL-LABEL: mgp.model @trait_constant_shared_lanes
// CONSTMODEL-COUNT-1: mgp.buffer.constant
mo.graph @trait_constant_shared_lanes() -> !mo.tensor<[2, 3], f32> {
  %legacy_cst = mo.constant {value = #M.dense_array<3., 3., 3., 3., 3., 3.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %trait_cst = mo.constant {value = #M.dense_array<3., 3., 3., 3., 3., 3.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %0 = mo.custom {symbol = "trait_mixed_legacy_fused", target = "cpu", device = #M.device_ref<"cpu", 0>}(%legacy_cst, %trait_cst) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// Equal constants with different trait contracts share one capture; execute
// still builds both wrappers.
// FOLD-LABEL: mo.graph @trait_constant_distinct_contracts
// FOLD: mogg.kernel()
// FOLD-COUNT-1: mo.constant
// CONSTMODEL-LABEL: mgp.model @trait_constant_distinct_contracts
// CONSTMODEL-COUNT-1: mgp.buffer.constant
// EMITCONST: #   custom__trait_dense_and_fused_inputs
// EMITCONST: [[SHARED_CAP:var___[0-9]+]] = _MoggTensorBufferCapture
// EMITCONST: _MoggInputLoadFusionTensor[type_of([[SHARED_CAP]]).dtype
// EMITCONST: _MoggDenseTensorArg[type_of([[SHARED_CAP]]).dtype
// EMITCONST: TraitDenseAndFusedInputs.execute
mo.graph @trait_constant_distinct_contracts() -> !mo.tensor<[2, 3], f32> {
  %dense_cst = mo.constant {value = #M.dense_array<4., 4., 4., 4., 4., 4.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %fused_cst = mo.constant {value = #M.dense_array<4., 4., 4., 4., 4., 4.> : tensor<2x3xf32>} : !mo.tensor<[2, 3], f32>
  %0 = mo.custom {symbol = "trait_dense_and_fused_inputs", target = "cpu", device = #M.device_ref<"cpu", 0>}(%dense_cst, %fused_cst) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}
