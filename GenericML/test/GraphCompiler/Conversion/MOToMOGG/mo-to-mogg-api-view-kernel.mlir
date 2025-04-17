// RUN: mo-opt %s --mo-to-mogg | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.
// RUN: MOGG_USE_EXP_KERNELS=1 mo-opt %s --mo-to-mogg

// Check mo-to-mogg correctly lowers a view KernelAPI

// CHECK: mogg.kernel
// CHECK: %[[OUT:.*]] = mogg.output_placeholders
// CHECK: %[[KOUT:.*]] = "mogg.to_kgen"(%[[OUT]])
// COM: Build shape
// CHECK: %[[SHAPE:.*]] = pop.array.create
// CHECK: %[[KSHAPE:.*]] = kgen.rebind %[[SHAPE]]
// COM: Call to the mojo copy implementation
// CHECK: kgen.call @"tensor_internal::managed_tensor_slice::view_copy_impl{{.+}}"<{{.+}}>(%[[KOUT]], %{{.+}})
// CHECK: mogg.output %[[OUT]]

mo.graph @broadcast_to(%arg0: !mo.tensor<[1, 8], f32>) -> !mo.tensor<[6, 3, 8], f32> {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[1, 8], f32> -> !mo.tensor<[6, 3, 8], f32>
  mo.output %0 : !mo.tensor<[6, 3, 8], f32>
}
