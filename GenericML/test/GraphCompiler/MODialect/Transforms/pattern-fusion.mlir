// RUN: mo-opt %s --mo-pattern-fusion  | FileCheck %s
// TODO: Under TensorFlow only because that's the default framework label for the graph compiler.

// CHECK: mo.graph @split_zero_axis(%[[INPUT:.*]]:
// CHECK: %[[RES:.*]]:2 = mogg.split(%[[INPUT]])
// CHECK: mo.output %[[RES]]#0, %[[RES]]#1

mo.graph @split_zero_axis(
  %input: !mo.tensor<[5, 4], f32>
) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>) {
  %ax = mo.constant {value = #M.dense_array<0> : tensor<si32>} : !mo.tensor<[], si32>
  %split_sizes = mo.constant {value = #M.dense_array<2, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %res:2 = mo.split[%ax: !mo.tensor<[], si32>](%input, %split_sizes) : (
    !mo.tensor<[5, 4], f32>, !mo.tensor<[2], si64>
  ) -> (!mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>)

  mo.output %res#0, %res#1 : !mo.tensor<[2, 4], f32>, !mo.tensor<[3, 4], f32>
}

// CHECK-LABEL: mo.graph @concat
// CHECK-NEXT: %[[IN1:.*]] = mo.static.reshape
// CHECK-NEXT: %[[IN2:.*]] = mo.static.reshape
// CHECK-NEXT: %[[OUT:.*]] = mogg.concat(%[[IN1]], %[[IN2]])
// CHECK-NEXT: mo.output %[[OUT]]

mo.graph @concat(%in: !mo.tensor<[2, 8], f32>) -> (!mo.tensor<[4, 2, 4], f32>) {
  %0 = mo.constant {value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32>
  %1 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %2 = mo.static.reshape(%in) : !mo.tensor<[2, 8], f32> -> !mo.tensor<[2, 2, 4], f32>
  %3 = mo.concat[%0 : !mo.tensor<[], si32>] (%1, %2) : (!mo.tensor<[2, 2, 4], f32>, !mo.tensor<[2, 2, 4], f32>) -> (!mo.tensor<[4, 2, 4], f32>)
  mo.output %3 : !mo.tensor<[4, 2, 4], f32>
}

// CHECK-LABEL: mo.graph @concat_gpu
// CHECK-NEXT: %[[IN1:.*]] = mo.static.reshape
// CHECK-NEXT: %[[IN2:.*]] = mo.static.reshape
// CHECK-NEXT: %[[OUT:.*]] = mogg.concat(%[[IN1]], %[[IN2]])
// CHECK-NEXT: mo.output %[[OUT]] : !mo.tensor<[4, 2, 4], f32, cuda:0>

mo.graph @concat_gpu(%arg0: !mo.tensor<[2, 8], f32, cuda:0>) -> !mo.tensor<[4, 2, 4], f32, cuda:0> attributes {device = #M.device_ref<"cuda", 0>} {
  %0 = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0> : tensor<1xsi32>} : !mo.tensor<[], si32, cpu:0>
  %1 = mo.static.reshape(%arg0) {device = #M.device_ref<"cuda", 0>} : !mo.tensor<[2, 8], f32, cuda:0> -> !mo.tensor<[2, 2, 4], f32, cuda:0>
  %2 = mo.static.reshape(%arg0) {device = #M.device_ref<"cuda", 0>} : !mo.tensor<[2, 8], f32, cuda:0> -> !mo.tensor<[2, 2, 4], f32, cuda:0>
  %3 = mo.concat[%0 : !mo.tensor<[], si32, cpu:0>] (%1, %2) {device = #M.device_ref<"cuda", 0>} : (!mo.tensor<[2, 2, 4], f32, cuda:0>, !mo.tensor<[2, 2, 4], f32, cuda:0>) -> !mo.tensor<[4, 2, 4], f32, cuda:0>
  mo.output %3 : !mo.tensor<[4, 2, 4], f32, cuda:0>
}

// CHECK-LABEL: mo.graph @batch_matmul_transpose_b_cpu
// CHECK: mo.batch_matmul(%{{.*}}, %{{.*}}) {transpose_b = true}
mo.graph @batch_matmul_transpose_b_cpu(%input: !mo.tensor<[10, 100, 1000], f32>, %weight: !mo.tensor<[10, 10000, 1000], f32>) -> !mo.tensor<[10, 100, 10000], f32> {
  %perm = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 2, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32, cpu:0>
  %transpose_weight = mo.transpose(%weight, %perm): (!mo.tensor<[10, 10000, 1000], f32>, !mo.tensor<[3], si32, cpu:0>) -> !mo.tensor<[10, 1000, 10000], f32>
  %res = mo.batch_matmul(%input, %transpose_weight): (!mo.tensor<[10, 100, 1000], f32>, !mo.tensor<[10, 1000, 10000], f32>) -> !mo.tensor<[10, 100, 10000], f32>
  mo.output %res : !mo.tensor<[10, 100, 10000], f32>
}

// `transpose_b=True` on GPU for batch matmul is not supported rn.
// CHECK-LABEL: mo.graph @batch_matmul_transpose_b_cuda_disabled
// CHECK-NOT: {transpose_b = true}
mo.graph @batch_matmul_transpose_b_cuda_disabled(%input: !mo.tensor<[10, 100, 1000], f32, cuda:0>, %weight: !mo.tensor<[10, 10000, 1000], f32, cuda:0>) -> !mo.tensor<[10, 100, 10000], f32, cuda:0> attributes {device = #M.device_ref<"cuda", 0>} {
  %perm = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 2, 1> : tensor<3xsi32>} : !mo.tensor<[3], si32, cpu:0>
  %transpose_weight = mo.transpose(%weight, %perm): (!mo.tensor<[10, 10000, 1000], f32, cuda:0>, !mo.tensor<[3], si32, cpu:0>) -> !mo.tensor<[10, 1000, 10000], f32, cuda:0>
  %res = mo.batch_matmul(%input, %transpose_weight): (!mo.tensor<[10, 100, 1000], f32, cuda:0>, !mo.tensor<[10, 1000, 10000], f32, cuda:0>) -> !mo.tensor<[10, 100, 10000], f32, cuda:0>
  mo.output %res : !mo.tensor<[10, 100, 10000], f32, cuda:0>
}
