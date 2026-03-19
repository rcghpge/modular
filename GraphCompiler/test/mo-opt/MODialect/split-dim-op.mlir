// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK-LABEL: @split_one_rhs
// CHECK: mo.add_singleton_dim[2] ({{.*}}) : (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 9, 1, 16], f32>
mo.graph @split_one_rhs(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 9, 1, 16], f32> {
    %out = mo.split_dim[1](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 9, 1, 16], f32>
    mo.output %out: !mo.tensor<[4, 9, 1, 16], f32>
}

// CHECK-LABEL: @split_one_lhs
// CHECK: mo.add_singleton_dim[1] ({{.*}}) : (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 1, 9, 16], f32>
mo.graph @split_one_lhs(%arg0: !mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 1, 9, 16], f32> {
    %out = mo.split_dim[1](%arg0): (!mo.tensor<[4, 9, 16], f32>) -> !mo.tensor<[4, 1, 9, 16], f32>
    mo.output %out: !mo.tensor<[4, 1, 9, 16], f32>
}
