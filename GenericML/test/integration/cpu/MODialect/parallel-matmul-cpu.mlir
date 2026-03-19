// RUN: mo-opt %s --gc-pipeline=constant-fold=false --mlir-print-ir-after=mogg-to-mgp -o %t.mlir 2> %t.mgp
// RUN: mt --execute --result-output-style=full %t.mlir | FileCheck %s
// RUN: FileCheck %s --check-prefix=CHECK-MGP < %t.mgp


// Input 1: [[1, 0], [0, -1]]
// Input 2: [[0, 1], [-1, 0]]

// --- Matmul alone: non-elementwise kernel in parallel body ---
// matmul(A, A):
//   [[1,0],[0,-1]] * [[1,0],[0,-1]] = [[1,0],[0,1]]
//   [[0,1],[-1,0]] * [[0,1],[-1,0]] = [[-1,0],[0,-1]]
mo.graph @parallel_matmul() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 0.0, 0.0, -1.0> : tensor<2x2xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<0.0, 1.0, -1.0, 0.0> : tensor<2x2xf32>} : !mo.tensor<[2, 2], f32>
  %res:2 = mo.parallel %arg0 in (%a : !mo.tensor<[2, 2], f32>,
                                  %b : !mo.tensor<[2, 2], f32>)
         -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
    %mm = mo.matmul(%arg0, %arg0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    mo.yield %mm : !mo.tensor<[2, 2], f32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK: 'parallel_matmul' returned tensor<2x2xf32>
// CHECK-SAME: [1.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, 1.{{0*}}e+00]
// CHECK: , tensor<2x2xf32>
// CHECK-SAME: [-1.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, -1.{{0*}}e+00]


// --- Matmul + relu: epilogue fusion may merge into 1 kernel ---
// relu(matmul(A, A)):
//   relu([[1,0],[0,1]]) = [[1,0],[0,1]]
//   relu([[-1,0],[0,-1]]) = [[0,0],[0,0]]
mo.graph @parallel_matmul_relu() -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
  %a = mo.constant {value = #M.dense_array<1.0, 0.0, 0.0, -1.0> : tensor<2x2xf32>} : !mo.tensor<[2, 2], f32>
  %b = mo.constant {value = #M.dense_array<0.0, 1.0, -1.0, 0.0> : tensor<2x2xf32>} : !mo.tensor<[2, 2], f32>
  %res:2 = mo.parallel %arg0 in (%a : !mo.tensor<[2, 2], f32>,
                                  %b : !mo.tensor<[2, 2], f32>)
         -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) {
    %mm = mo.matmul(%arg0, %arg0) : (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) -> !mo.tensor<[2, 2], f32>
    %relu = mo.relu(%mm) : !mo.tensor<[2, 2], f32>
    mo.yield %relu : !mo.tensor<[2, 2], f32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK: 'parallel_matmul_relu' returned tensor<2x2xf32>
// CHECK-SAME: [1.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, 1.{{0*}}e+00]
// CHECK: , tensor<2x2xf32>
// CHECK-SAME: [0.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00, 0.{{0*}}e+00]

// CHECK-MGP: mogg.generic.compiled.region
// CHECK-MGP:   mogg.call.stub.async @stub_{{[0-9]+}}({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.call.stub.async @stub_{{[0-9]+}}({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.async.join
// CHECK-MGP:   mogg.async.ready

// CHECK-MGP: mogg.generic.compiled.region
// CHECK-MGP:   mogg.call.stub.async @stub_{{[0-9]+}}({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.call.stub.async @stub_{{[0-9]+}}({{.*}}) -> !mogg.task_handle
// CHECK-MGP:   mogg.async.join
// CHECK-MGP:   mogg.async.ready
