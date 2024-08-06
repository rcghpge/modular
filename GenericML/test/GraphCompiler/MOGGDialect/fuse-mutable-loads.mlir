// RUN: mo-opt --mo-to-mogg --fuse-mutable-loads %s | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_lhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: [[out:%.*]]:2 = mogg.kernel ([[buffer]], [[rhs]], [[ch0]])
// CHECK-SAME: ([[kbuffer:%.*]]: !mo.buffer<[10, 10], f32>, [[krhs:%.*]]: !mo.tensor<[10, 10], f32>{{.*}}!mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
// CHECK: [[op:%.*]]:2 = mogg.output_placeholders{{.*}}!mo.chain
// CHECK-DAG: mogg.call "simd_load"([[kbuffer]]
// CHECK-DAG: mogg.call "simd_load"([[krhs]]
// CHECK: mogg.output [[op]]#0, [[op]]#1
// CHECK: inputFusionInterface = ["", "{{.*}}", ""]{{.*}}outputFusionInterface = ["{{.*}}", ""]
// CHECK: mo.mutable.store([[out]]#1, [[buffer]], [[out]]#0)
mo.graph @single_add_buffer_lhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outChain, %outTensor = mo.mutable.load(%0, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%outTensor, %rhs) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %buffer, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: [[out:%.*]]:2 = mogg.kernel ([[rhs]], [[buffer]], [[ch0]])
// CHECK-SAME: ([[krhs:%.*]]: !mo.tensor<[10, 10], f32>, [[kbuffer:%.*]]: !mo.buffer<[10, 10], f32>{{.*}}!mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
// CHECK: [[op:%.*]]:2 = mogg.output_placeholders{{.*}}!mo.chain
// CHECK-DAG: mogg.call "simd_load"([[kbuffer]]
// CHECK-DAG: mogg.call "simd_load"([[krhs]]
// CHECK: mogg.output [[op]]#0, [[op]]#1
// CHECK: inputFusionInterface = ["{{.*}}", "", ""]{{.*}}outputFusionInterface = ["{{.*}}", ""]
// CHECK: mo.mutable.store([[out]]#1, [[buffer]], [[out]]#0)
mo.graph @single_add_buffer_rhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outChain, %outTensor = mo.mutable.load(%0, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %buffer, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  mo.output
}
