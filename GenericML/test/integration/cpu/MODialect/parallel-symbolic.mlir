// RUN: mo-opt %s --gc-pipeline=constant-fold=false --mlir-print-ir-after=mogg-to-mgp -o %t.mlir 2> %t.mgp
// RUN: FileCheck %s < %t.mgp

// Symbolic-dim parallel: relu with a named dimension parameter.
mo.graph @parallel_symbolic<N>(
    %arg0: !mo.tensor<[N, 3], si32>,
    %arg1: !mo.tensor<[N, 3], si32>)
    -> (!mo.tensor<[N, 3], si32>, !mo.tensor<[N, 3], si32>) {
  %res:2 = mo.parallel %x in (%arg0 : !mo.tensor<[N, 3], si32>,
                              %arg1 : !mo.tensor<[N, 3], si32>)
         -> (!mo.tensor<[N, 3], si32>, !mo.tensor<[N, 3], si32>) {
    %1 = mo.relu(%x) : !mo.tensor<[N, 3], si32>
    mo.yield %1 : !mo.tensor<[N, 3], si32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[N, 3], si32>, !mo.tensor<[N, 3], si32>
}

// CHECK: mogg.generic.compiled.region
// CHECK:   mogg.call.stub.async @stub_{{[0-9]+}}
// CHECK:   mogg.call.stub.async @stub_{{[0-9]+}}
// CHECK:   mogg.async.join
// CHECK:   mogg.async.ready


// Symbolic-dim parallel with multiple kernels: add(x, x) + relu.
mo.graph @parallel_symbolic_add_relu<N>(
    %arg0: !mo.tensor<[N, 3], f32>,
    %arg1: !mo.tensor<[N, 3], f32>)
    -> (!mo.tensor<[N, 3], f32>, !mo.tensor<[N, 3], f32>) {
  %res:2 = mo.parallel %x in (%arg0 : !mo.tensor<[N, 3], f32>,
                              %arg1 : !mo.tensor<[N, 3], f32>)
         -> (!mo.tensor<[N, 3], f32>, !mo.tensor<[N, 3], f32>) {
    %added = mo.add(%x, %x) : !mo.tensor<[N, 3], f32>
    %relu = mo.relu(%added) : !mo.tensor<[N, 3], f32>
    mo.yield %relu : !mo.tensor<[N, 3], f32>
  }
  mo.output %res#0, %res#1 : !mo.tensor<[N, 3], f32>, !mo.tensor<[N, 3], f32>
}

// CHECK: mogg.generic.compiled.region
// CHECK:   mogg.call.stub.async @stub_{{[0-9]+}}
// CHECK:   mogg.call.stub.async @stub_{{[0-9]+}}
// CHECK:   mogg.async.join
// CHECK:   mogg.async.ready
