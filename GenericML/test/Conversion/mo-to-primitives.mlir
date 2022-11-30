// RUN: mop-opt %s --mo-to-primitives | FileCheck %s

// CHECK:      mop.model @empty : mop init {
// CHECK:        %0 = grt.chain.create
// CHECK:        %1 = mop.context.create 0
// CHECK:        bef.output %0, %1 : !bef.chain, !mop.context
// CHECK:      } execute (
// CHECK-SAME:   %arg0: !bef.chain, %arg1: !mop.context, %arg2: !mop.cpu.tensor<2xf32>, %arg3: !mop.cpu.tensor<5x1xf32>
// CHECK-SAME: ) -> (!bef.chain, !mop.cpu.tensor<2xf32>, !mop.cpu.tensor<5x1xf32>, !mop.cpu.tensor<5x1xf32>) {
// CHECK:        bef.output %arg0, %arg2, %arg3, %arg3 : !bef.chain, !mop.cpu.tensor<2xf32>, !mop.cpu.tensor<5x1xf32>, !mop.cpu.tensor<5x1xf32>
// CHECK:      }
mo.graph @empty(%arg0: !mo.tensor<2xf32>, %arg1: !mo.tensor<5x1xf32>) -> (!mo.tensor<2xf32>, !mo.tensor<5x1xf32>, !mo.tensor<5x1xf32>) {
  mo.output %arg0, %arg1, %arg1 : !mo.tensor<2xf32>, !mo.tensor<5x1xf32>, !mo.tensor<5x1xf32>
}
