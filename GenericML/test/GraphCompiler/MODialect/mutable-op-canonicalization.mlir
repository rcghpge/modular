// RUN: mo-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: all_same_buffer
// CHECK-SAME: [[buf:%.*]]: !mo.buffer
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: [[ch1:%.*]], [[tensor:%.*]] = mo.mutable.load([[ch0]], [[buf]])
// CHECK-NOT: mo.mutable.load
// CHECK-NOT: mo.mutable.load
// CHECK: mo.negative([[tensor]])
// CHECK: mo.abs([[tensor]])
// CHECK: mo.relu([[tensor]])
mo.graph @all_same_buffer(%buf: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) 
{
  %0 = mo.chain.create()
  %outChain, %outTensor = mo.mutable.load(%0, %buf) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %outChain_0, %outTensor_1 = mo.mutable.load(%outChain, %buf) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %outChain_2, %outTensor_3 = mo.mutable.load(%outChain_0, %buf) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %1 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
  %2 = mo.abs(%outTensor_1) : !mo.tensor<[2, 2], f32>
  %3 = mo.relu(%outTensor_3) : !mo.tensor<[2, 2], f32>
  mo.output %1, %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}

// CHECK-LABEL: different_buffer_in_between
// CHECK-SAME: [[buf0:%.*]]: !mo.buffer<{{.*}}>, [[buf1:%.*]]: !mo.buffer<{{.*}}>
// CHECK: [[ch0:%.*]] = mo.chain.create
// CHECK: [[ch1:%.*]], [[tensor0:%.*]] = mo.mutable.load([[ch0]], [[buf0]])
// CHECK: [[ch2:%.*]], [[tensor1:%.*]] = mo.mutable.load([[ch1]], [[buf1]])
// CHECK: [[ch3:%.*]], [[tensor2:%.*]] = mo.mutable.load([[ch2]], [[buf0]])
// CHECK: mo.negative([[tensor0]])
// CHECK: mo.abs([[tensor1]])
// CHECK: mo.relu([[tensor2]])
mo.graph @different_buffer_in_between(%buf1: !mo.buffer<[2, 2], f32>, %buf2: !mo.buffer<[2, 2], f32>) -> (!mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>) 
{
  %0 = mo.chain.create()
  %outChain, %outTensor = mo.mutable.load(%0, %buf1) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %outChain_0, %outTensor_1 = mo.mutable.load(%outChain, %buf2) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %outChain_2, %outTensor_3 = mo.mutable.load(%outChain_0, %buf1) : (!mo.chain, !mo.buffer<[2, 2], f32>) -> (!mo.chain, !mo.tensor<[2, 2], f32>)
  %1 = mo.negative(%outTensor) : !mo.tensor<[2, 2], f32>
  %2 = mo.abs(%outTensor_1) : !mo.tensor<[2, 2], f32>
  %3 = mo.relu(%outTensor_3) : !mo.tensor<[2, 2], f32>
  mo.output %1, %2, %3 : !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>, !mo.tensor<[2, 2], f32>
}
