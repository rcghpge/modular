// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --gc-pipeline-pre-jit="dump-mojo-text=true" -o /dev/null 2>&1 | FileCheck %s

// MAP-lane counterpart of codegen-inline-broadcast-constant-scalars.mlir: the
// broadcast constant is emitted as a literal operand of the elementwise call,
// splatted to the closure's SIMD width so it works in any operand position,
// and the kernel takes one input pointer rather than two -- the constant
// buffer is never loaded from, or even passed in.

// CHECK: def {{[A-Za-z0-9_]+_map(_[0-9]+)?}}([[IN:[^:]*]]: OpaquePointer[MutAnyOrigin], [[OUT:[^:]*]]: OpaquePointer[MutAnyOrigin], [[CTX:.*]]: {{.*}}DeviceContext) raises:
// CHECK:   var [[MTS0:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, False, extensibility.IO.Input, {{[0-9]+}}]([[IN]]{{.*}})
// CHECK:   var [[MTS1:.*]] = {{.*}}mogg_tensor_init[DType.float32, 2, True, extensibility.IO.Output, {{[0-9]+}}]([[OUT]]{{.*}})
// CHECK:   def [[LAMBDA:lambda___[0-9]+]][width : Int, element_alignment : Int = 1](idx: layout.Coord) {var}:
// CHECK:     var [[XV:.*]] = [[MTS0]].load[width = width, element_alignment = element_alignment](idx)
// CHECK:     var [[RES:.*]] = builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]([[XV]], SIMD[DType.float32, width](2.0))
// CHECK:     [[MTS1]].store[width = width, element_alignment = element_alignment](idx, rebind[SIMD[type_of([[MTS1]]).dtype, width]]([[RES]]))
// The constant is inlined, so nothing reads a zero-strided view of it.
// CHECK-NOT: mogg_tensor_create_broadcast

mo.graph @add_scalar_constant(%arg0: !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32> {
  %s = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[2, 2], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[2, 2], f32>
  mo.output %r : !mo.tensor<[2, 2], f32>
}
