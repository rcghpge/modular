// RUN: mo-opt --canonicalize %s | FileCheck --enable-var-scope %s

// A trivial parallel (body is just yield %arg) should be folded away,
// replacing all results with the corresponding inputs.

// CHECK-LABEL: mo.graph @fold_trivial_parallel
// CHECK-SAME: (%[[A:[^ ]*]]: !mo.tensor<[3], f32>, %[[B:[^ ]*]]: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
// CHECK-NEXT: mo.output %[[A]], %[[B]] : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
mo.graph @fold_trivial_parallel(%a: !mo.tensor<[3], f32>, %b: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %res:2 = mo.parallel %arg in (%a : !mo.tensor<[3], f32>,
                                 %b : !mo.tensor<[3], f32>)
         -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
    mo.yield %arg : !mo.tensor<[3], f32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// A non-trivial parallel body should NOT be folded.

// CHECK-LABEL: mo.graph @keep_nontrivial_parallel
// CHECK: mo.parallel
mo.graph @keep_nontrivial_parallel(%a: !mo.tensor<[3], f32>, %b: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %res:2 = mo.parallel %arg in (%a : !mo.tensor<[3], f32>,
                                 %b : !mo.tensor<[3], f32>)
         -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32>
    mo.yield %r : !mo.tensor<[3], f32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}
