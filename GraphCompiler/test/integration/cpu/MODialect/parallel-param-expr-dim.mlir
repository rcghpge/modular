// RUN: not mo-opt %s --gc-pipeline-pre-jit 2>&1 | FileCheck %s

// Regression test: `mo.parallel` with parameter-expression-form dims
// (e.g. `div(D0, 2)`) in tensor types must not reach MGP lowering with
// SSA dim operands silently dropped.
//
// HoistParameterExpressionsPass intentionally skips ops inside `mo.parallel`
// bodies (HoistParameterExpressionsPass.cpp:97-98), so a tensor type with a
// `KGEN::ParamOperatorAttr` dim that flows through a parallel body reaches
// MOGGToMGP unrewritten. The kernel-operand-mapping helper
// `KernelOperandsMapping::getTensorDynamicDimensions` used to silently
// drop the dim (it only resolved `KGEN::ParamDeclRefAttr`), leaving the
// resulting kernel stub without the SSA dim operand `EmitMojo` needs.
// In production builds (NDEBUG) this surfaced as `std::bad_alloc`.

!t_in = !mo.tensor<[D0, 3], f32>
!t = !mo.tensor<[div(D0, 2), 3], f32>
!b = !mo.bundle<[!t, !t]>

mo.graph @parallel_param_expr_dim<D0>(%arg0: !t_in, %arg1: !t_in) -> (!t, !t) {
  %r0 = mo.rebind(%arg0) : !t_in -> !t
  %r1 = mo.rebind(%arg1) : !t_in -> !t
  %bundle = mo.tensor.bundle(%r0, %r1) : (!t, !t) -> (!b)
  %res_bundle = mo.parallel (%x) in (%bundle : !b) -> (!b) {
    %1 = mo.relu(%x) : !t
    mo.yield %1 : !t
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!b) -> (!t, !t)
  mo.output %res_0, %res_1 : !t, !t
}

// CHECK: getTensorDynamicDimensions: unsupported dim attr kind
// CHECK-SAME: Expected IntegerAttr or KGEN::ParamDeclRefAttr
