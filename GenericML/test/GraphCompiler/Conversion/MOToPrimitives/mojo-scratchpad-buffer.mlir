// RUN: mo-opt -mo-to-mogg -fuse-elementwise -fuse-prologues-epilogues -mo-to-primitives %s | FileCheck %s

// CHECK-LABEL: mogg.kernel.stub @stub_0
// CHECK:         mogg.bind
// CHECK:         mogg.tensor.fused
// CHECK:         mogg.call.execute["mo.matmul"]
// CHECK:       }
// CHECK:       mogg.generic.compiled.region
// CHECK:         mogg.tensor.__init__(%{{.*}}) {exclusive = true, io = 1 : i32, mut = false}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {exclusive = true, io = 1 : i32, mut = false}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {exclusive = true, io = 1 : i32, mut = false}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {exclusive = true, io = 0 : i32, mut = true}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {exclusive = true, io = 0 : i32, mut = true}
// CHECK:         mogg.call.stub @stub_0
// CHECK:         mogg.async.ready : !mef.chain
// CHECK:         mogg.output
// CHECK:       }

mo.graph @matmul_cast(%a : !mo.tensor<[4, 4], f32>, %b: !mo.tensor<[4, 4], f32>, %c: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], si32> {
  %0 = mo.matmul(%a, %b) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %1 = mo.add(%0, %c) : !mo.tensor<[4, 4], f32>
  %2 = mo.cast(%1) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], si32>
  mo.output %2 : !mo.tensor<[4, 4], si32>
}
