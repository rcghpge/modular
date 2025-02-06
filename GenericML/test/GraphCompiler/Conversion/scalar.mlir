// RUN: mo-opt %s --mo-to-mogg | FileCheck %s

// CHECK-LABEL: mo.graph @mean
// CHECK: "mogg.to_kgen"
// CHECK: "mogg.to_kgen"
// CHECK: %[[TENSOR:.*]] = "mogg.to_kgen"
// CHECK: %[[SCALAR:.*]] = kgen.call @"{{.*}}MOGGKernelAPI::get_scalar_from_managed_tensor_slice{{.*}}"{{.*}}(%[[TENSOR:.*]])
// CHECK: kgen.call @"{{.*}}MOGGKernelAPI::Mean::execute{{.*}}"{{.*}}({{.*}}%[[SCALAR:.*]]{{.*}})

mo.graph @mean(%input: !mo.tensor<[100, 50], f32>) -> (!mo.tensor<[100, 1], f32>) {
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %output = mo.mean(%input, %axis) : (!mo.tensor<[100, 50], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[100, 1], f32>
  mo.output %output : !mo.tensor<[100, 1], f32>
}
