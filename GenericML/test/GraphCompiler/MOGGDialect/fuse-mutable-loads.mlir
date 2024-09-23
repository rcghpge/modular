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

// CHECK-LABEL: mo.graph @multiple_chain_users(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: mo.chain.create([[ch0]])
// CHECK: [[out:%.*]]:2 = mogg.kernel ([[rhs]], [[buffer]], [[ch0]])
// CHECK-DAG: mo.mutable.store([[out]]#1, [[buffer]], [[out]]#0)
// CHECK-DAG: mo.chain.create([[out]]#1)
mo.graph @multiple_chain_users(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  mo.chain.create(%0) : !mo.chain
  %outChain, %outTensor = mo.mutable.load(%0, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%rhs, %outTensor) : !mo.tensor<[10, 10], f32>
  mo.chain.create(%outChain) : !mo.chain
  %2 = mo.mutable.store(%outChain, %buffer, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @same_kernel_multiple_fusions(
// CHECK-SAME: [[buf1:%.*]]: !mo.buffer<[10, 10], f32>, [[buf2:%.*]]: !mo.buffer<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mo.chain.create([[ch0]], [[ch0]])
// CHECK: mogg.kernel ([[buf1]], [[buf2]], [[ch1]])
mo.graph @same_kernel_multiple_fusions(%buf1: !mo.buffer<[10, 10], f32>, %buf2: !mo.buffer<[10, 10], f32>) {

  %ch0 = mo.chain.create()

  %ch1, %vbuf = mo.mutable.load(%ch0, %buf1) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %ch2, %vbuf2 = mo.mutable.load(%ch0, %buf2) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %res = mo.add(%vbuf, %vbuf2) : !mo.tensor<[10, 10], f32>
  %chfinal = mo.mutable.store(%ch1, %buf2, %res) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> (!mo.chain)

  mo.output
}
