// RUN: mo-opt %s --mo-to-mogg --fuse-elementwise | FileCheck %s

// CHECK:  mogg.bind
// CHECK:    mogg.call.elementwise["mo.add"]
// CHECK:    mogg.call.elementwise["mo.relu"]
// CHECK:    mogg.output
// CHECK:  }
// CHECK:  mogg.call.foreach
mo.graph @test(%add_lhs: !mo.tensor<[3, 3], f64>, %add_rhs: !mo.tensor<[3, 3], f64>) -> !mo.tensor<[3, 3], f64> {
    %add = mo.add(%add_lhs, %add_rhs) : !mo.tensor<[3, 3], f64>
    %relu = mo.relu(%add) : !mo.tensor<[3, 3], f64>
    mo.output %relu: !mo.tensor<[3, 3], f64>
}
