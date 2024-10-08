// RUN: mo-opt %s --mo-to-mogg --canonicalize --split-input-file | FileCheck %s


mo.graph @concat(%in_0 : !mo.tensor<[10], si32>, %in_1: !mo.tensor<[10], si32>) -> (!mo.tensor<[20], si32>) {
    %axis = mo.constant {value = #M.dense_array<0> : tensor<si64>} : !mo.tensor<[], si64>
    // CHECK: %[[tuple:.*]] = mogg.tuple.create
    // CHECK: mogg.call "mo.concat" [@normal_op_output_0_fn_counter_3](%{{.*}}, %[[tuple]], %{{.*}})
    %concat = mo.concat[%axis : !mo.tensor<[], si64>] (%in_0, %in_1) : (!mo.tensor<[10], si32>, !mo.tensor<[10], si32>) -> !mo.tensor<[20], si32>
    mo.output %concat : !mo.tensor<[20], si32>
}
