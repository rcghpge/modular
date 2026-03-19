// RUN: mo-opt %s --canonicalize | FileCheck %s

// Verify that mo.parallel with tupled extra inputs parses and round-trips.
// The body receives two block arguments: the primary input and the extra input.

// CHECK-LABEL: mo.graph @parallel_with_extra_inputs
mo.graph @parallel_with_extra_inputs(
    %w0: !mo.tensor<[3], f32>,
    %w1: !mo.tensor<[3], f32>
) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %in0 = mo.constant {value = #M.dense_array<1.0, 2.0, 3.0> : tensor<3xf32>} : !mo.tensor<[3], f32>
  %in1 = mo.constant {value = #M.dense_array<4.0, 5.0, 6.0> : tensor<3xf32>} : !mo.tensor<[3], f32>

  // CHECK: mo.parallel
  // CHECK-SAME: (%arg2, %arg3) in
  %res:2 = mo.parallel (%arg, %extra) in
      ((%in0, %w0) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>),
       (%in1, %w1) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)) {
    %sum = mo.add(%arg, %extra) : !mo.tensor<[3], f32>
    mo.yield %sum : !mo.tensor<[3], f32>
  }

  mo.output %res#0, %res#1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}
