// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Verify async dispatch structure at the MGP stage.
// RUN: mo-opt %s --mo-to-mogg --mgp-mo-to-primitives | FileCheck %s --check-prefix=CHECK-MGP

// Execute two relu ops
mo.graph @sequential_relu() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant {value = #M.dense_array<1, -2, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %const2 = mo.constant {value = #M.dense_array<-9, 8, -7> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %res1 = mo.relu(%const1) : !mo.tensor<[3], si32>
  %res2 = mo.relu(%const2) : !mo.tensor<[3], si32>
  mo.output %res1, %res2 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// CHECK: 'sequential_relu' returned tensor<3xsi32> [1, 0, 3]
// CHECK: , tensor<3xsi32> [0, 8, 0]


// Test for mo.parallel op
mo.graph @parallel_op() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant {value = #M.dense_array<1, -2, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %const2 = mo.constant {value = #M.dense_array<-9, 8, -7> : tensor<3xsi32>} : !mo.tensor<[3], si32>
  %res:2 = mo.parallel %arg0 in (%const1 : !mo.tensor<[3], si32>,
                                  %const2 : !mo.tensor<[3], si32>)
         -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
    %1 = mo.relu(%arg0) : !mo.tensor<[3], si32>
    mo.yield %1 : !mo.tensor<[3], si32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// Verify the parallel op lowers to async dispatch at the MGP stage.
// CHECK-MGP: mogg.kernel.stub @parallel_stub_0
// CHECK-MGP: mogg.generic.compiled.region
// CHECK-MGP:   mogg.call.stub.async @parallel_stub_0({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.call.stub.async @parallel_stub_0({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.async.join({{.*}}) : (!mogg.task_handle, !mogg.task_handle)
// CHECK-MGP:   mogg.async.ready
