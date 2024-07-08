// RUN: mo-opt --mo-to-mogg --fuse-mutable-ops %s | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_lhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: [[ch2:%.*]] = mogg.kernel ([[buffer]], [[rhs]], [[ch0]])
// CHECK-SAME: ([[kbuffer:%.*]]: !mo.buffer<[10, 10], f32>, [[krhs:%.*]]: !mo.tensor<[10, 10], f32>{{.*}}!mo.chain) -> !mo.chain
// CHECK: [[kChain:%.*]] = mogg.output_placeholders : !mo.chain
// CHECK: mogg.call "simd_load"([[kbuffer]]
// CHECK: mogg.call "simd_load"([[krhs]]
// CHECK: mogg.call "simd_store"([[kbuffer]]
// CHECK: mogg.call "elementwise" [@elemwise_compute_lambda]([[kbuffer]])
// CHECK: mogg.output [[kChain]] : !mo.chain
// CHECK: inputFusionInterface = ["input_0_fn", "input_1_fn", ""]
// CHECK-SAME: outputFusionInterface = [""]
// CHECK-SAME: summary = "inplace_mo.add
// CHECK: mo.mutable.load([[ch2]], [[buffer]])
mo.graph @single_add_buffer_lhs(%buffer: !mo.buffer<[10, 10], f32>, %rhs: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create(%buffer) : !mo.buffer<[10, 10], f32>
  %outChain, %outTensor = mo.mutable.load(%0, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mogg.kernel (%outTensor, %rhs) : (%arg2: !mo.tensor<[10, 10], f32>, %arg3: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32>{
    %3 = mogg.output_placeholders : !mo.tensor<[10, 10], f32>
    mogg.lambda @input_0_fn = (%arg4: !mogg.indices) -> !mogg.simd<f32>{
      %4 = mogg.call "simd_load"(%arg2, %arg4) : !mo.tensor<[10, 10], f32>, !mogg.indices -> !mogg.simd<f32>
      mogg.output %4 : !mogg.simd<f32>
    }
    mogg.lambda @input_1_fn = (%arg4: !mogg.indices) -> !mogg.simd<f32>{
      %4 = mogg.call "simd_load"(%arg3, %arg4) : !mo.tensor<[10, 10], f32>, !mogg.indices -> !mogg.simd<f32>
      mogg.output %4 : !mogg.simd<f32>
    }
    mogg.lambda @output_0_fn = (%arg4: !mogg.indices, %arg5: !mogg.simd<f32>){
      mogg.call "simd_store"(%3, %arg4, %arg5) : !mo.tensor<[10, 10], f32>, !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.lambda @elemwise_compute_lambda = (%arg4: !mogg.indices){
      %4 = mogg.invoke @input_0_fn(%arg4) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
      %5 = mogg.invoke @input_1_fn(%arg4) {readOnly = true} : !mogg.indices -> !mogg.simd<f32>
      %6 = mogg.call "mo.add"(%4, %5) : !mogg.simd<f32>, !mogg.simd<f32> -> !mogg.simd<f32>
      mogg.invoke @output_0_fn(%arg4, %6) : !mogg.indices, !mogg.simd<f32>
      mogg.output
    }
    mogg.call "elementwise" [@elemwise_compute_lambda](%3) {single_thread_blocking_override = false} : !mo.tensor<[10, 10], f32>
    mogg.output %3 : !mo.tensor<[10, 10], f32>
  } {fusionBodyLambda = "elemwise_compute_lambda", inputFusionInterface = ["input_0_fn", "input_1_fn"], isAsync = true, outputFusionInterface = ["output_0_fn"], summary = "mo.add:10"}
  %2 = mo.mutable.store(%outChain, %buffer, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  %outChain_0, %outTensor_1 = mo.mutable.load(%2, %buffer) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  mo.output %outTensor_1 : !mo.tensor<[10, 10], f32>
}

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[lhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: [[ch2:%.*]] = mogg.kernel ([[lhs]], [[buffer]], [[ch0]])
// CHECK-SAME: ([[klhs:%.*]]: !mo.tensor<[10, 10], f32>, [[kbuffer:%.*]]: !mo.buffer<[10, 10], f32>{{.*}}!mo.chain) -> !mo.chain
// CHECK: [[kChain:%.*]] = mogg.output_placeholders : !mo.chain
// CHECK: mogg.call "simd_load"([[klhs]]
// CHECK: mogg.call "simd_load"([[kbuffer]]
// CHECK: mogg.call "simd_store"([[kbuffer]]
// CHECK: mogg.call "elementwise" [@{{.*}}]([[kbuffer]])
// CHECK: mogg.output [[kChain]] : !mo.chain
// CHECK: inputFusionInterface = ["input_0_fn{{.*}}", "input_1_fn{{.*}}", ""]
// CHECK-SAME: outputFusionInterface = [""]
// CHECK-SAME: summary = "inplace_mo.add
// CHECK: mo.mutable.load([[ch2]], [[buffer]])
mo.graph @single_add_buffer_rhs(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[10, 10], f32>

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_0, %outTensor_1 = mo.mutable.load(%2, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  mo.output %outTensor_1 : !mo.tensor<[10, 10], f32>
}

// CHECK-LABEL: mo.graph @sequenced_add(
// CHECK-SAME: [[buffer:%.*]]: !mo.buffer<[10, 10], f32>, [[rhs:%.*]]: !mo.tensor<[10, 10], f32>
// CHECK: [[ch0:%.*]] = mo.chain.create([[buffer]])
// CHECK: [[ch2:%.*]] = mogg.kernel ([[buffer]], [[rhs]], [[ch0]])
// CHECK-SAME: ([[kbuffer:%.*]]: !mo.buffer<[10, 10], f32>, [[krhs:%.*]]: !mo.tensor<[10, 10], f32>{{.*}}!mo.chain) -> !mo.chain
// CHECK: [[kChain:%.*]] = mogg.output_placeholders : !mo.chain
// CHECK: mogg.call "simd_load"([[kbuffer]]
// CHECK: mogg.call "simd_load"([[krhs]]
// CHECK: mogg.call "simd_store"([[kbuffer]]
// CHECK: mogg.call "elementwise" [@{{.*}}]([[kbuffer]])
// CHECK: mogg.output [[kChain]] : !mo.chain
// CHECK: inputFusionInterface = ["input_0_fn{{.*}}", "input_1_fn{{.*}}", ""]
// CHECK-SAME: outputFusionInterface = [""]
// CHECK-SAME: summary = "inplace_mo.add
// CHECK: [[ch3:%.*]] = mogg.kernel ([[buffer]], [[rhs]], [[ch2]])
// CHECK-SAME: ([[kbuffer:%.*]]: !mo.buffer<[10, 10], f32>, [[krhs:%.*]]: !mo.tensor<[10, 10], f32>{{.*}}!mo.chain) -> !mo.chain
// CHECK: [[kChain:%.*]] = mogg.output_placeholders : !mo.chain
// CHECK: mogg.call "simd_load"([[kbuffer]]
// CHECK: mogg.call "simd_load"([[krhs]]
// CHECK: mogg.call "simd_store"([[kbuffer]]
// CHECK: mogg.call "elementwise" [@{{.*}}]([[kbuffer]])
// CHECK: mogg.output [[kChain]] : !mo.chain
// CHECK: inputFusionInterface = ["input_0_fn{{.*}}", "input_1_fn{{.*}}", ""]
// CHECK-SAME: outputFusionInterface = [""]
// CHECK-SAME: summary = "inplace_mo.add
// CHECK: mo.mutable.load([[ch3]], [[buffer]])
mo.graph @sequenced_add(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create(%arg0) : !mo.buffer<[10, 10], f32>

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_0, %outTensor_1 = mo.mutable.load(%2, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %3 = mo.add(%outTensor_1, %arg1) : !mo.tensor<[10, 10], f32>
  %4 = mo.mutable.store(%outChain_0, %arg0, %3) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_2, %outTensor_3 = mo.mutable.load(%4, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)

  mo.output %outTensor_3 : !mo.tensor<[10, 10], f32>
}
