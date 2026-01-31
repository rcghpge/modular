// RUN: mo-opt %s --canonicalize

// Test for mo.parallel op using custom assembly format
mo.graph @parallel_op() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant {value = #M.dense_array<1, -2, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %const2 = mo.constant {value = #M.dense_array<-9, 8, -7> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %res:2 = mo.parallel %arg0 in (%const1, %const2 : !mo.tensor<[3], si32>) {
    %1 = mo.relu(%arg0) : !mo.tensor<[3], si32>
    mo.yield %1 : !mo.tensor<[3], si32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}
