// RUN: mo-opt %s --mo-to-mgp=constant-fold=false -o %t.mlir
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s

// Verify async dispatch structure at the MGP stage.
// RUN: mo-opt %s --mo-to-mogg --mgp-mo-to-primitives | FileCheck %s --check-prefix=CHECK-MGP

// Test for mo.parallel op on GPU
mo.graph @parallel_gpu_op() -> (!mo.tensor<[3], si32>, !mo.tensor<[3], si32>) {
  %const1 = mo.constant {value = #M.dense_array<1, -2, 3> : tensor<3xsi32>} : !mo.tensor<[3], si32, gpu:0>
  %const2 = mo.constant {value = #M.dense_array<-9, 8, -7> : tensor<3xsi32>} : !mo.tensor<[3], si32, gpu:0>
  %res:2 = mo.parallel %arg0 in (%const1, %const2 : !mo.tensor<[3], si32, gpu:0>) {
    %1 = mo.relu(%arg0) : !mo.tensor<[3], si32, gpu:0>
    mo.yield %1 : !mo.tensor<[3], si32, gpu:0>
  }
  // Transfer GPU results back to CPU for output
  %ch0 = mo.chain.create()
  %res0_host, %ch1 = mo.transfer[%ch0] %res#0 : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  %res1_host, %ch2 = mo.transfer[%ch1] %res#1 : !mo.tensor<[3], si32, gpu:0> to <"cpu", 0>
  mo.output %res0_host, %res1_host : !mo.tensor<[3], si32>, !mo.tensor<[3], si32>
}

// CHECK: 'parallel_gpu_op' returned tensor<3xsi32> [1, 0, 3]
// CHECK: , tensor<3xsi32> [0, 8, 0]

// Verify the parallel op lowers to async dispatch at the MGP stage.
// CHECK-MGP: mogg.kernel.stub @parallel_stub_0
// CHECK-MGP: mogg.generic.compiled.region
// CHECK-MGP:   mogg.call.stub.async @parallel_stub_0({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.call.stub.async @parallel_stub_0({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.async.join({{.*}}) : (!mogg.task_handle, !mogg.task_handle)
// CHECK-MGP:   mogg.async.ready
