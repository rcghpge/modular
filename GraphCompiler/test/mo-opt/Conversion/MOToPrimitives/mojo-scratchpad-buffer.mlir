// RUN: mo-opt --mogg-import-packages -mo-to-mogg -mogg-fuse-elementwise -mogg-fuse-epilogues -mogg-to-mgp %s | FileCheck %s

// CHECK-LABEL: mogg.kernel.stub @Epilogue_matmul_Elementwise_add_{{[0-9a-f]+}}_mogg
// CHECK:         mogg.bind
// CHECK:         mogg.tensor.fused
// CHECK:         mogg.call.execute["{{.*}}.Matmul"]
// CHECK:       }
// CHECK:       mogg.generic.compiled.region
// CHECK:         mogg.tensor.__init__(%{{.*}}) {io = 1 : i32, mut = false}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {io = 1 : i32, mut = false}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {io = 1 : i32, mut = false}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {io = 0 : i32, mut = true}
// CHECK:         mogg.tensor.__init__(%{{.*}}) {io = 0 : i32, mut = true}
// CHECK:         mogg.call.stub @Epilogue_matmul_Elementwise_add_{{[0-9a-f]+}}_mogg
// CHECK:         mogg.async.ready : !mef.chain
// CHECK:         mogg.output
// CHECK:       }

mo.graph @matmul_cast(%a : !mo.tensor<[4, 4], f32>, %b: !mo.tensor<[4, 4], f32>, %c: !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], si32> {
  %0 = mo.matmul(%a, %b) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %1 = mo.add(%0, %c) : !mo.tensor<[4, 4], f32>
  %2 = mo.cast(%1) : (!mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], si32>
  mo.output %2 : !mo.tensor<[4, 4], si32>
}
