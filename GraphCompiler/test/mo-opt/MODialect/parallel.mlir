// RUN: mo-opt --canonicalize %s | FileCheck --enable-var-scope %s

// A trivial parallel (body is just yield %arg) should be folded away,
// replacing all results with the corresponding inputs.

// CHECK-LABEL: mo.graph @fold_trivial_parallel
// CHECK-SAME: (%[[A:[^ ]*]]: !mo.tensor<[3], f32>, %[[B:[^ ]*]]: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
// CHECK-NEXT: mo.output %[[A]], %[[B]] : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
mo.graph @fold_trivial_parallel(%a: !mo.tensor<[3], f32>, %b: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %res_bundle = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    mo.yield %arg : !mo.tensor<[3], f32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}

// A non-trivial parallel body should NOT be folded.

// CHECK-LABEL: mo.graph @keep_nontrivial_parallel
// CHECK: mo.parallel
mo.graph @keep_nontrivial_parallel(%a: !mo.tensor<[3], f32>, %b: !mo.tensor<[3], f32>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) {
  %bundle = mo.tensor.bundle(%a, %b) : (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>) -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
  %res_bundle = mo.parallel (%arg) in (%bundle : !mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>)
         -> (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) {
    %r = mo.relu(%arg) : !mo.tensor<[3], f32>
    mo.yield %r : !mo.tensor<[3], f32>
  }
  %res_0, %res_1 = mo.tensor.unbundle(%res_bundle) : (!mo.bundle<[!mo.tensor<[3], f32>, !mo.tensor<[3], f32>]>) -> (!mo.tensor<[3], f32>, !mo.tensor<[3], f32>)
  mo.output %res_0, %res_1 : !mo.tensor<[3], f32>, !mo.tensor<[3], f32>
}
