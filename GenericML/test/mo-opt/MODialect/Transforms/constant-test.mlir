// RUN: mo-opt %s --mo-constant-fold | FileCheck %s


// CHECK: mo.graph @gather_param_no_fold<[[SH0:.*]]: !mosh.ape -> [[SH1:.*]]: !mosh.ape>(
mo.graph @gather_param_no_fold<Sh0: !mosh.ape -> Sh1: !mosh.ape>(
    // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[SH0]], si32>) -> !mo.tensor<[[SH1]], si32> {
    %input: !mo.tensor<Sh0, si32>) -> !mo.tensor<Sh1, si32> {
  // CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK-DAG: %[[INDICES:.*]] = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %indices = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.gather<() -> [[SH2:.*]]: !mosh.ape>(%[[INPUT]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<Sh0, si32>, !mo.tensor<[0], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, si32>
  %1 = mo.gather<() -> Sh2: !mosh.ape>(%input, %indices, %axis) : (!mo.tensor<Sh0, si32>, !mo.tensor<[0], si64>, !mo.tensor<[], si64>) -> !mo.tensor<Sh2, si32>
  // CHECK-NEXT: mo.output<:!mosh.ape [[SH2]]> %[[RES]] : !mo.tensor<[[SH2]], si32>
  mo.output<:!mosh.ape Sh2> %1 : !mo.tensor<Sh2, si32>
}
