// RUN: mo-opt --mogg-import-packages --mo-to-mogg --mogg-fuse-elementwise --canonicalize --cse --mogg-fuse-views --canonicalize --cse --mogg-fuse-prologues --mogg-fuse-epilogues --mogg-inline-broadcast-constant-scalars %s | FileCheck %s

// A fully-broadcast compile-time constant scalar operand of an elementwise
// lambda is inlined as a `mogg.constant.scalar` literal instead of being loaded
// through a `StaticBroadcastTo` view. The constant is also dropped as a kernel
// operand.

// A rank-0 f32 constant broadcast to the full shape is inlined.
mo.graph @scalar_add_f32(%arg0: !mo.tensor<[10, 4, 8], f32>) -> !mo.tensor<[10, 4, 8], f32> {
  %s = mo.constant {value = #M.dense_array<2.000000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[10, 4, 8], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[10, 4, 8], f32>
  mo.output %r : !mo.tensor<[10, 4, 8], f32>
}
// CHECK-LABEL: mo.graph @scalar_add_f32
// The kernel takes only the tensor operand; the constant is not passed in.
// CHECK: mogg.kernel(%{{.*}}: !mo.tensor<[10, 4, 8], f32>) -> !mo.tensor<[10, 4, 8], f32>
// CHECK-NOT: mogg.call.view
// CHECK: mogg.constant.scalar {value = 2.000000e+00 : f32} : !mo.scalar<f32>
// CHECK: mogg.bind (%{{.*}}) elemwise_lambda (%{{.*}}: f32, %{{.*}}: !pop.array<3, index>) {
// CHECK: mogg.call.elementwise["{{.*}}.Add"] (%{{.*}}, %{{.*}}) {{.*}} : f32, !mo.scalar<f32> -> f32

// A rank-0 si32 constant broadcast to the full shape is inlined.
mo.graph @scalar_add_si32(%arg0: !mo.tensor<[6], si32>) -> !mo.tensor<[6], si32> {
  %s = mo.constant {value = #M.dense_array<3> : tensor<si32>} : !mo.tensor<[], si32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], si32> -> !mo.tensor<[6], si32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[6], si32>
  mo.output %r : !mo.tensor<[6], si32>
}
// CHECK-LABEL: mo.graph @scalar_add_si32
// CHECK: mogg.kernel(%{{.*}}: !mo.tensor<[6], si32>) -> !mo.tensor<[6], si32>
// CHECK-NOT: mogg.call.view
// CHECK: mogg.constant.scalar {value = 3 : si32} : !mo.scalar<si32>
// CHECK: mogg.call.elementwise["{{.*}}.Add"] (%{{.*}}, %{{.*}}) {{.*}} : si32, !mo.scalar<si32> -> si32

// A broadcast scalar constant reached through an intervening `mo.transfer`
// (as `ops.constant(...).to(device)` lowers) is inlined too.
mo.graph @scalar_add_transferred_const(%arg0: !mo.tensor<[6], f32>) -> !mo.tensor<[6], f32> {
  %ch0 = mo.chain.create()
  %s = mo.constant {value = #M.dense_array<7.500000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %st, %ch1 = mo.transfer[%ch0] %s : !mo.tensor<[], f32> to <"cpu", 0> {alwaysElideSameDeviceCopy = false}
  %b = mo.static.broadcast_to(%st) : !mo.tensor<[], f32> -> !mo.tensor<[6], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[6], f32>
  mo.output %r : !mo.tensor<[6], f32>
}
// CHECK-LABEL: mo.graph @scalar_add_transferred_const
// CHECK: mogg.kernel(%{{.*}}: !mo.tensor<[6], f32>) -> !mo.tensor<[6], f32>
// CHECK-NOT: mogg.call.view
// CHECK: mogg.constant.scalar {value = 7.500000e-01 : f32} : !mo.scalar<f32>
// CHECK: mogg.call.elementwise["{{.*}}.Add"] (%{{.*}}, %{{.*}}) {{.*}} : f32, !mo.scalar<f32> -> f32

// A broadcast constant captured as a fusible operand of an input (prologue)
// lambda is inlined too. Here `mul(arg0, broadcast(0.5))` fuses as a prologue
// into the softmax kernel, so the broadcast is the second fusible operand.
mo.graph @softmax_mul_const_prologue(%arg0: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
  %s = mo.constant {value = #M.dense_array<5.000000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[2, 3], f32>
  %mul = mo.mul(%arg0, %b) : !mo.tensor<[2, 3], f32>
  %sm = mo.reduce.softmax(%mul) {axis = 1 : index} : (!mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
  mo.output %sm : !mo.tensor<[2, 3], f32>
}
// CHECK-LABEL: mo.graph @softmax_mul_const_prologue
// The constant tensor is dropped; the kernel takes only the runtime input.
// CHECK: mogg.kernel(%{{.*}}: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
// CHECK-NOT: mogg.call.view
// CHECK: mogg.constant.scalar {value = 5.000000e-01 : f32} : !mo.scalar<f32>
// CHECK: mogg.bind (%{{.*}}) input_lambda (%{{.*}}: f32, %{{.*}}: !pop.array<2, index>) {
// CHECK: mogg.call.elementwise["{{.*}}.Mul"] (%{{.*}}, %{{.*}}) {{.*}} : f32, !mo.scalar<f32> -> f32

// A broadcast constant captured by an output (epilogue) lambda is inlined too.
// Here `mul(matmul(A, B), broadcast(0.25))` fuses as an epilogue on the matmul
// kernel, so the broadcast is read via a `mogg.tensor.load` inside the body
// rather than being a fusible operand.
mo.graph @matmul_mul_const_epilogue(%A: !mo.tensor<[5, 4], f32>, %B: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 4], f32> {
  %s = mo.constant {value = #M.dense_array<2.500000e-01> : tensor<f32>} : !mo.tensor<[], f32>
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[5, 4], f32>
  %mm = mo.matmul(%A, %B) : (!mo.tensor<[5, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 4], f32>
  %mul = mo.mul(%mm, %b) : !mo.tensor<[5, 4], f32>
  mo.output %mul : !mo.tensor<[5, 4], f32>
}
// CHECK-LABEL: mo.graph @matmul_mul_const_epilogue
// The constant tensor is dropped; the kernel takes only the matmul inputs.
// CHECK: mogg.kernel(%{{.*}}: !mo.tensor<[5, 4], f32>, %{{.*}}: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 4], f32>
// The captured zero-strided load is gone, replaced by an inlined scalar.
// CHECK-NOT: mogg.tensor.load
// CHECK: mogg.constant.scalar {value = 2.500000e-01 : f32} : !mo.scalar<f32>
// CHECK: mogg.bind (%{{.*}}) output_lambda (%{{.*}}: f32, %{{.*}}: !pop.array<2, index>) {
// CHECK: mogg.call.elementwise["{{.*}}.Mul"] (%{{.*}}, %{{.*}}) {{.*}} : f32, !mo.scalar<f32> -> f32

// A runtime (non-constant) scalar captured by an epilogue lambda is NOT
// inlined: the broadcast source is a kernel operand rather than a constant, so
// the captured `mogg.tensor.load` stays.
mo.graph @matmul_mul_runtime_epilogue(%A: !mo.tensor<[5, 4], f32>, %B: !mo.tensor<[4, 4], f32>, %s: !mo.tensor<[], f32>) -> !mo.tensor<[5, 4], f32> {
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], f32> -> !mo.tensor<[5, 4], f32>
  %mm = mo.matmul(%A, %B) : (!mo.tensor<[5, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[5, 4], f32>
  %mul = mo.mul(%mm, %b) : !mo.tensor<[5, 4], f32>
  mo.output %mul : !mo.tensor<[5, 4], f32>
}
// CHECK-LABEL: mo.graph @matmul_mul_runtime_epilogue
// CHECK-NOT: mogg.constant.scalar
// CHECK: mogg.tensor.load

// A runtime (non-constant) scalar broadcast is NOT inlined: it stays a view.
mo.graph @scalar_add_runtime(%arg0: !mo.tensor<[6], si32>, %s: !mo.tensor<[], si32>) -> !mo.tensor<[6], si32> {
  %b = mo.static.broadcast_to(%s) : !mo.tensor<[], si32> -> !mo.tensor<[6], si32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[6], si32>
  mo.output %r : !mo.tensor<[6], si32>
}
// CHECK-LABEL: mo.graph @scalar_add_runtime
// CHECK-NOT: mogg.constant.scalar
// CHECK: mogg.call.view["{{.*}}StaticBroadcastTo"]

// A partial (vector) broadcast of a constant is NOT a scalar: it stays a view.
mo.graph @vector_add(%arg0: !mo.tensor<[2, 4], f32>) -> !mo.tensor<[2, 4], f32> {
  %v = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0, 4.0> : tensor<4xf32>} : !mo.tensor<[4], f32>
  %b = mo.static.broadcast_to(%v) : !mo.tensor<[4], f32> -> !mo.tensor<[2, 4], f32>
  %r = mo.add(%arg0, %b) : !mo.tensor<[2, 4], f32>
  mo.output %r : !mo.tensor<[2, 4], f32>
}
// CHECK-LABEL: mo.graph @vector_add
// CHECK: mogg.call.view["{{.*}}StaticBroadcastTo"]
// CHECK-NOT: mogg.constant.scalar
