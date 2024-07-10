// RUN: mo-opt --mo-to-mgp %s | FileCheck %s

// CHECK: } execute ({{.*}}[[BUFFER:%.*]]: !mgp.tensor<2x2xf32>
// CHECK: mef.output{{.*}}[[BUFFER]]
mo.graph @buffer_output(%buffer: !mo.buffer<[2, 2], f32>) -> !mo.buffer<[2, 2], f32>
{
  mo.output %buffer : !mo.buffer<[2, 2], f32>
}
