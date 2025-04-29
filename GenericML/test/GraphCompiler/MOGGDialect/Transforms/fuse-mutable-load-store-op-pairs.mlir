// RUN: mo-opt --infer-layouts --mo-to-mogg --fuse-elementwise --fuse-in-out-lambdas --fuse-mutable-loads %s | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
// CHECK: mo.mutable.load([[ch1]]
mo.graph @single_add_buffer_rhs(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_0, %outTensor_1 = mo.mutable.load(%2, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  mo.output %outTensor_1 : !mo.tensor<[10, 10], f32>
}

// CHECK-LABEL: mo.graph @sequenced_add
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
// CHECK: mo.mutable.load([[ch1]]
mo.graph @sequenced_add(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %3 = mo.add(%1, %arg1) : !mo.tensor<[10, 10], f32>
  %4 = mo.mutable.store(%outChain, %arg0, %3) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %outChain_2, %outTensor_3 = mo.mutable.load(%4, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)

  mo.output %outTensor_3 : !mo.tensor<[10, 10], f32>
}

// In this case store fusion is not possible as add's output has a use other than the store
// CHECK-LABEL: mo.graph @no_valid_fusion
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]]:2 = mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}"
// CHECK: mogg.kernel ({{.*}}[[ch1]]#1)
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
mo.graph @no_valid_fusion(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %3 = mo.sub(%1, %arg1) : !mo.tensor<[10, 10], f32>

  mo.output %3 : !mo.tensor<[10, 10], f32>
}

// CHECK-LABEL: mo.graph @store_chain_comes_from_wrong_producer
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]]:2 = mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}"
// CHECK: mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
mo.graph @store_chain_comes_from_wrong_producer(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create()

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%0, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  mo.output
}

// CHECK-LABEL: mo.graph @store_to_fuse_has_sibling
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]]:2 = mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}"
// CHECK: [[ch2:%.*]] = mogg.kernel ({{.*}}[[ch1]]#1)
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
// CHECK: [[ch3:%.*]] = mogg.kernel ({{.*}}[[ch1]]#1)
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
mo.graph @store_to_fuse_has_sibling(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create()

  %outChain, %outTensor = mo.mutable.load(%0, %arg0) : (!mo.chain, !mo.buffer<[10, 10], f32>) -> (!mo.chain, !mo.tensor<[10, 10], f32>)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%outChain, %arg0, %1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain

  %3 = mo.mutable.store(%outChain, %arg0, %arg1) : (!mo.chain, !mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @load_matmul_add_store
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.matmul:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
mo.graph @load_matmul_add_store(%buf0: !mo.buffer<[4, 4], f32>, %arg1: !mo.tensor<[4, 4], f32>, %arg2: !mo.tensor<[4, 4], f32>) -> !mo.buffer<[4, 4], f32> {
  %ch0 = mo.chain.create()

  %ch1, %loaded_tensor = mo.mutable.load(%ch0, %buf0) : (!mo.chain, !mo.buffer<[4, 4], f32>) -> (!mo.chain, !mo.tensor<[4, 4], f32>)
  %1 = mo.matmul(%loaded_tensor, %arg1) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %2 = mo.add(%1, %arg2) : !mo.tensor<[4, 4], f32>
  %ch2 = mo.mutable.store(%ch1, %buf0, %2) : (!mo.chain, !mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> (!mo.chain)

  mo.output %buf0 : !mo.buffer<[4, 4], f32>
}

// CHECK-LABEL: mo.graph @load_matmul_store_twice
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mogg.kernel ({{.*}}[[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.matmul:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
// CHECK: [[ch2:%.*]] = mogg.kernel ({{.*}}[[ch1]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.matmul:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
mo.graph @load_matmul_store_twice(%buf0: !mo.buffer<[4, 4], f32>, %arg1: !mo.tensor<[4, 4], f32>, %arg2: !mo.tensor<[4, 4], f32>) -> !mo.buffer<[4, 4], f32> {
  %ch0 = mo.chain.create()

  // First sequence: load + matmul + store
  %ch1, %loaded_tensor1 = mo.mutable.load(%ch0, %buf0) : (!mo.chain, !mo.buffer<[4, 4], f32>) -> (!mo.chain, !mo.tensor<[4, 4], f32>)
  %1 = mo.matmul(%loaded_tensor1, %arg1) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %ch2 = mo.mutable.store(%ch1, %buf0, %1) : (!mo.chain, !mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> (!mo.chain)

  // Second sequence: load + matmul + store (operating on the same buffer)
  %ch3, %loaded_tensor2 = mo.mutable.load(%ch2, %buf0) : (!mo.chain, !mo.buffer<[4, 4], f32>) -> (!mo.chain, !mo.tensor<[4, 4], f32>)
  %2 = mo.matmul(%loaded_tensor2, %arg2) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %ch4 = mo.mutable.store(%ch3, %buf0, %2) : (!mo.chain, !mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> (!mo.chain)

  mo.output %buf0 : !mo.buffer<[4, 4], f32>
}
