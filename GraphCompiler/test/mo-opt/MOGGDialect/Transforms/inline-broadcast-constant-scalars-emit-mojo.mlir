// RUN: mo-opt %s --mogg-import-packages -mo-to-mogg --mogg-simplify-kernel-operands -mogg-fuse-elementwise --canonicalize --cse -mogg-fuse-views --canonicalize --cse -mogg-fuse-epilogues -mogg-inline-broadcast-constant-scalars -mogg-to-mgp -mogg-outline-kernels -mogg-add-try-except-blocks -mogg-emit-mojo -mgp-dump-mojo 2>&1 | FileCheck %s

// The broadcast constant scalar is emitted as a literal operand to the
// elementwise call. No `StaticBroadcastTo.update_input_view` is emitted, so the
// scalar is never loaded from memory.

// The integral-valued f32 constant becomes a `2.0` literal (".0" appended so it
// lexes as a float), the fractional f32 constant a bare `0.5` literal, and the
// si32 constant a `3` literal, each splatted to a SIMD vector so they work as
// elementwise operands in any position. inf/nan have no Mojo float literal, so
// they are constructed via the numerics helpers.
// CHECK-DAG: builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]({{.*}}SIMD[DType.float32, {{.*}}](2.0))
// CHECK-DAG: builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]({{.*}}SIMD[DType.float32, {{.*}}](0.5))
// CHECK-DAG: builtin_kernels.elementwise.Add.elementwise[dtype = DType.int32]({{.*}}SIMD[DType.int32, {{.*}}](3))
// CHECK-DAG: builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]({{.*}}-std.utils.numerics.inf[DType.float32]())
// CHECK-DAG: builtin_kernels.elementwise.Add.elementwise[dtype = DType.float32]({{.*}}std.utils.numerics.nan[DType.float32]())
// A broadcast constant captured by an output (epilogue) lambda is inlined as a
// splatted literal, and the zero-strided managed-tensor-slice load is gone.
// CHECK-DAG: builtin_kernels.elementwise.Mul.elementwise[dtype = DType.float32]({{.*}}SIMD[DType.float32, {{.*}}](0.25))
// CHECK-NOT: update_input_view
// CHECK-NOT: simd_load_from_managed_tensor_slice

mo.graph @scalar_add_f32(%arg0: !mo.tensor<[10, 4, 8], f32>) -> !mo.tensor<[10, 4, 8], f32> {
  %s = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[10, 4, 8], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[10, 4, 8], f32>
  mo.output %r : !mo.tensor<[10, 4, 8], f32>
}

mo.graph @scalar_add_f32_fractional(%arg0: !mo.tensor<[10, 4, 8], f32>) -> !mo.tensor<[10, 4, 8], f32> {
  %s = mo.constant {value = #M.dense_array<5.000000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[10, 4, 8], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[10, 4, 8], f32>
  mo.output %r : !mo.tensor<[10, 4, 8], f32>
}

mo.graph @scalar_add_si32(%arg0: !mo.tensor<[6], si32>) -> !mo.tensor<[6], si32> {
  %s = mo.constant {value = #M.dense_array<3> : tensor<si32>} : !mo.tensor<[], si32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], si32> -> !mo.tensor<[6], si32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[6], si32>
  mo.output %r : !mo.tensor<[6], si32>
}

// -inf bit pattern (0xFF800000) -> -std.utils.numerics.inf[DType.float32]().
mo.graph @scalar_add_neg_inf(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %s = mo.constant {value = #M.dense_array<0xFF800000> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}

// nan bit pattern (0x7FC00000) -> std.utils.numerics.nan[DType.float32]().
mo.graph @scalar_add_nan(%arg0: !mo.tensor<[4], f32>) -> !mo.tensor<[4], f32> {
  %s = mo.constant {value = #M.dense_array<0x7FC00000> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[4], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[4], f32>
  mo.output %r : !mo.tensor<[4], f32>
}

// A 0.25 constant captured by the matmul epilogue (mo.mul) is inlined into the
// output lambda as `SIMD[..](0.25)` instead of a captured zero-strided load.
mo.graph @matmul_mul_const_epilogue(%A: !mo.tensor<[5, 4], f32>, %B: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 4], f32> {
  %s = mo.constant {value = #M.dense_array<2.500000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[5, 4], f32>
  %mm = mo.matmul(%A, %B) : (!mo.tensor<[5, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 4], f32>
  %mul = mo.mul(%mm, %b) : !mo.tensor<[5, 4], f32>
  mo.output %mul : !mo.tensor<[5, 4], f32>
}
