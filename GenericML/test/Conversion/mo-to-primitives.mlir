// RUN: mgp-opt %s --mo-to-primitives | FileCheck %s

// CHECK:      mgp.model @empty : mgp init {
// CHECK:        %0 = grt.chain.create
// CHECK:        %1 = mgp.context.create 0
// CHECK:        bef.output %0, %1 : !bef.chain, !mgp.context
// CHECK:      } execute (
// CHECK-SAME:   %arg0: !bef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<2xf32>, %arg3: !mgp.cpu.tensor<5x1xf32>
// CHECK-SAME: ) -> (!bef.chain, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<5x1xf32>, !mgp.cpu.tensor<5x1xf32>) {
// CHECK:        bef.output %arg0, %arg2, %arg3, %arg3 : !bef.chain, !mgp.cpu.tensor<2xf32>, !mgp.cpu.tensor<5x1xf32>, !mgp.cpu.tensor<5x1xf32>
// CHECK:      }
mo.graph @empty(%arg0: !mo.tensor<2xf32>, %arg1: !mo.tensor<5x1xf32>) -> (!mo.tensor<2xf32>, !mo.tensor<5x1xf32>, !mo.tensor<5x1xf32>) {
  mo.output %arg0, %arg1, %arg1 : !mo.tensor<2xf32>, !mo.tensor<5x1xf32>, !mo.tensor<5x1xf32>
}
