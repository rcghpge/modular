// TODO(GEX-1573): Convert the following sniff test to a complete test.

// RUN: mo-opt %s --mo-to-mogg | FileCheck %s

// Check mo-to-mogg correctly lowers a view KernelAPI

// CHECK: mogg.experimental.kernel
// CHECK: %[[OUT:.*]] = mogg.output_placeholders
// CHECK: %{{.*}} = mogg.tensor.shape
// CHECK: mogg.call.view["mo.static.broadcast_to"]
// CHECK: mogg.call.materialize

mo.graph @broadcast_to(%arg0: !mo.tensor<[1, 8], f32>) -> !mo.tensor<[6, 3, 8], f32> {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[1, 8], f32> -> !mo.tensor<[6, 3, 8], f32>
  mo.output %0 : !mo.tensor<[6, 3, 8], f32>
}
