// RUN: mo-opt %s --mo-constant-fold | FileCheck %s


// CHECK: mo.graph @gather_param_no_fold<[[D0:.*]], [[D1:.*]] -> [[D2:.*]], [[D3:.*]]>(
mo.graph @gather_param_no_fold<D0, D1 -> D2, D3>(
    // CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[[D0]], [[D1]]], si32>) -> !mo.tensor<[[[D2]], [[D3]]], si32> {
    %input: !mo.tensor<[D0, D1], si32>) -> !mo.tensor<[D2, D3], si32> {
  // CHECK-DAG: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK-DAG: %[[INDICES:.*]] = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
  %indices = mo.constant {value = #M.dense_array<> : !M.array<0xsi64>} : !mo.tensor<[0], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.gather<() -> [[D4:.*]], [[D5:.*]]>(%[[INPUT]], %[[INDICES]], %[[AXIS]]) : (!mo.tensor<[[[D0]], [[D1]]], si32>, !mo.tensor<[0], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[[[D4]], [[D5]]], si32>
  %1 = mo.gather<() -> D4, D5>(%input, %indices, %axis) : (!mo.tensor<[D0, D1], si32>, !mo.tensor<[0], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[D4, D5], si32>
  // CHECK-NEXT: mo.output<[[D4]], [[D5]]> %[[RES]] : !mo.tensor<[[[D4]], [[D5]]], si32>
  mo.output<D4, D5> %1 : !mo.tensor<[D4, D5], si32>
}
