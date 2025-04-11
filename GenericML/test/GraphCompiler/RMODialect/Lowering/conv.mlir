// RUN: mo-opt --legalize-rmo-operators --canonicalize %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @conv
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 32], f32>,
// CHECK:  %[[FILTER:.*]]: !mo.tensor<[2, 2, 32, 64], f32>)
// CHECK: -> !mo.tensor<[10, 4, 4, 64], f32>

mo.graph @conv(
  %input: !mo.tensor<[10, 5, 5, 32], f32>,
  %filter: !mo.tensor<[2, 2, 32, 64], f32>
) -> !mo.tensor<[10, 4, 4, 64], f32> {

  // CHECK-DAG: %[[GROUPS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[C0:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK: %[[OUT:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[C1]], dilations = %[[C1]], paddings = %[[C0]], num_groups = %[[GROUPS]]]
  %out = rmo.conv(%input, %filter) {
    strides = #mosh<ape[1, 1]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    num_groups = 1 : si64
  }: (!mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>) -> !mo.tensor<[10, 4, 4, 64], f32>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[10, 4, 4, 64], f32>
  mo.output %out : !mo.tensor<[10, 4, 4, 64], f32>
}

// -----

// CHECK-LABEL: mo.graph @conv_gpu
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 32], f32, gpu:0>,
// CHECK:  %[[FILTER:.*]]: !mo.tensor<[2, 2, 32, 64], f32, gpu:0>)
// CHECK: -> !mo.tensor<[10, 4, 4, 64], f32, gpu:0>

mo.graph @conv_gpu(
  %input: !mo.tensor<[10, 5, 5, 32], f32, gpu:0>,
  %filter: !mo.tensor<[2, 2, 32, 64], f32, gpu:0>
) -> !mo.tensor<[10, 4, 4, 64], f32, gpu:0> {

  // CHECK-DAG: %[[GROUPS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant {device = #M.device_ref<"gpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64, gpu:0>
  // CHECK-DAG: %[[C0:.*]] = mo.constant {device = #M.device_ref<"gpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64, gpu:0>
  // CHECK: %[[OUT:.*]] = mo.conv(%[[INPUT]], %[[FILTER]]) [strides = %[[C1]], dilations = %[[C1]], paddings = %[[C0]], num_groups = %[[GROUPS]]]
  %out = rmo.conv(%input, %filter) {
    strides = #mosh<ape[1, 1]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    num_groups = 1 : si64
  }: (!mo.tensor<[10, 5, 5, 32], f32, gpu:0>, !mo.tensor<[2, 2, 32, 64], f32, gpu:0>) -> !mo.tensor<[10, 4, 4, 64], f32, gpu:0>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[10, 4, 4, 64], f32, gpu:0>
  mo.output %out : !mo.tensor<[10, 4, 4, 64], f32, gpu:0>
}
