// RUN: MAX_GC_USE_ADV_FUSION=1 mo-opt %s --mogg-import-packages --mo-to-map --map-fuse-kernels | FileCheck %s

// A multi-use binary op over TWO large operands must NOT be duplicated: the Add
// is materialized into its own kernel and each consumer reads its result.
mo.graph @binaryAddLargeFanout(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>, %arg1: !mo.tensor<[4096, 2], f32, gpu:0>) -> (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %2 = mo.relu(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @binaryAddLargeFanout
// CHECK: %[[ADD:.*]] = map.iter.foreach
// CHECK: map.call.elementwise @{{.*}}.Add
// CHECK-NOT: map.call.elementwise @{{.*}}.Add
// CHECK: map.iter.foreach
// CHECK: inputs (%[[ADD]]
// CHECK: map.call.elementwise @{{.*}}.Tanh
// CHECK: map.iter.foreach
// CHECK: inputs (%[[ADD]]
// CHECK: map.call.elementwise @{{.*}}.ReLU

// A multi-use UNARY op has at most one large operand, so re-reading it on
// duplication is not a net loss. It is still cloned into each consumer's fused
// subgraph (Exp appears in both kernels), preserving the existing fusion win.
mo.graph @unaryExpLargeFanout(%arg0: !mo.tensor<[4096, 2], f32, gpu:0>) -> (!mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>) {
  %0 = mo.exp(%arg0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %1 = mo.tanh(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  %2 = mo.relu(%0) : !mo.tensor<[4096, 2], f32, gpu:0>
  mo.output %1, %2 : !mo.tensor<[4096, 2], f32, gpu:0>, !mo.tensor<[4096, 2], f32, gpu:0>
}

// CHECK-LABEL: mo.graph @unaryExpLargeFanout
// CHECK: map.iter.foreach
// CHECK: map.call.elementwise @{{.*}}.Exp
// CHECK: map.call.elementwise @{{.*}}.Tanh
// CHECK: map.iter.foreach
// CHECK: map.call.elementwise @{{.*}}.Exp
// CHECK: map.call.elementwise @{{.*}}.ReLU
