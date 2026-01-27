// RUN: mo-opt --mo-infer-layouts --mo-to-mogg --mogg-fuse-elementwise --mogg-fuse-prologues-epilogues --mogg-fuse-mutable-loads %s | FileCheck %s

// CHECK-LABEL: mo.graph @single_add_buffer_rhs(
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mogg.kernel{{.*}}chain([[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
// CHECK: mo.mutable.load
mo.graph @single_add_buffer_rhs(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%arg0, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  %outTensor_1, %outChain_0 = mo.mutable.load(%arg0, %2) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  mo.output %outTensor_1 : !mo.tensor<[10, 10], f32>
}

// CHECK-LABEL: mo.graph @sequenced_add
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mogg.kernel{{.*}}chain([[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
// CHECK: mo.mutable.load
mo.graph @sequenced_add(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %3 = mo.add(%1, %arg1) : !mo.tensor<[10, 10], f32>
  %4 = mo.mutable.store(%arg0, %3, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  %outTensor_3, %outChain_2 = mo.mutable.load(%arg0, %4) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)

  mo.output %outTensor_3 : !mo.tensor<[10, 10], f32>
}

// In this case store fusion is not possible as add's output has a use other than the store
// CHECK-LABEL: mo.graph @no_valid_fusion
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]]:2 = mogg.kernel{{.*}}chain([[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}"
// CHECK: mogg.kernel{{.*}}chain([[ch1]]#1)
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
mo.graph @no_valid_fusion(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) -> !mo.tensor<[10, 10], f32> {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%outTensor, %arg1) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%arg0, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  %3 = mo.sub(%1, %arg1) : !mo.tensor<[10, 10], f32>

  mo.output %3 : !mo.tensor<[10, 10], f32>
}

// CHECK-LABEL: mo.graph @store_chain_comes_from_wrong_producer
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]]:2 = mogg.kernel{{.*}}chain([[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}"
// CHECK: mogg.kernel{{.*}}chain([[ch0]])
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
mo.graph @store_chain_comes_from_wrong_producer(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%arg0, %1, %0) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  mo.output
}

// CHECK-LABEL: mo.graph @store_to_fuse_has_sibling
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]]:2 = mogg.kernel{{.*}}chain([[ch0]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.add:{{[0-9]+}}"
// CHECK: [[ch2:%.*]] = mogg.kernel{{.*}}chain([[ch1]]#1)
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
// CHECK: [[ch3:%.*]] = mogg.kernel{{.*}}chain([[ch1]]#1)
// CHECK: summary = "mo.mutable.store:{{[0-9]+}}"
mo.graph @store_to_fuse_has_sibling(%arg0: !mo.buffer<[10, 10], f32>, %arg1: !mo.tensor<[10, 10], f32>) {
  %0 = mo.chain.create()

  %outTensor, %outChain = mo.mutable.load(%arg0, %0) : (!mo.buffer<[10, 10], f32>, !mo.chain) -> (!mo.tensor<[10, 10], f32>, !mo.chain)
  %1 = mo.add(%arg1, %outTensor) : !mo.tensor<[10, 10], f32>
  %2 = mo.mutable.store(%arg0, %1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain

  %3 = mo.mutable.store(%arg0, %arg1, %outChain) : (!mo.buffer<[10, 10], f32>, !mo.tensor<[10, 10], f32>, !mo.chain) -> !mo.chain
  mo.output
}

// CHECK-LABEL: mo.graph @load_matmul_add_store
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[outTensor:%.*]], [[outChain:%.*]] = mo.mutable.load(%arg0, [[ch0]])
// CHECK: [[ch1:%.*]] = mogg.kernel{{.*}}chain([[outChain]])
// CHECK: summary = "mo.matmul:{{[0-9]+}}+mo.add:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
mo.graph @load_matmul_add_store(%buf0: !mo.buffer<[4, 4], f32>, %arg1: !mo.tensor<[4, 4], f32>, %arg2: !mo.tensor<[4, 4], f32>) -> !mo.buffer<[4, 4], f32> {
  %ch0 = mo.chain.create()

  %loaded_tensor, %ch1 = mo.mutable.load(%buf0, %ch0) : (!mo.buffer<[4, 4], f32>, !mo.chain) -> (!mo.tensor<[4, 4], f32>, !mo.chain)
  %1 = mo.matmul(%loaded_tensor, %arg1) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %2 = mo.add(%1, %arg2) : !mo.tensor<[4, 4], f32>
  %ch2 = mo.mutable.store(%buf0, %2, %ch1) : (!mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>, !mo.chain) -> (!mo.chain)

  mo.output %buf0 : !mo.buffer<[4, 4], f32>
}

// CHECK-LABEL: mo.graph @load_matmul_store_twice
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[outTensor1:%.*]], [[outChain1:%.*]] = mo.mutable.load(%arg0, [[ch0]])
// CHECK: [[ch1:%.*]] = mogg.kernel{{.*}}chain([[outChain1]])
// CHECK: summary = "mo.matmul:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
// CHECK: [[outTensor2:%.*]], [[outChain2:%.*]] = mo.mutable.load(%arg0, [[ch1]])
// CHECK: [[ch2:%.*]] = mogg.kernel{{.*}}chain([[outChain2]])
// CHECK: summary = "mo.matmul:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
mo.graph @load_matmul_store_twice(%buf0: !mo.buffer<[4, 4], f32>, %arg1: !mo.tensor<[4, 4], f32>, %arg2: !mo.tensor<[4, 4], f32>) -> !mo.buffer<[4, 4], f32> {
  %ch0 = mo.chain.create()

  // First sequence: load + matmul + store
  %loaded_tensor1, %ch1 = mo.mutable.load(%buf0, %ch0) : (!mo.buffer<[4, 4], f32>, !mo.chain) -> (!mo.tensor<[4, 4], f32>, !mo.chain)
  %1 = mo.matmul(%loaded_tensor1, %arg1) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %ch2 = mo.mutable.store(%buf0, %1, %ch1) : (!mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>, !mo.chain) -> (!mo.chain)

  // Second sequence: load + matmul + store (operating on the same buffer)
  %loaded_tensor2, %ch3 = mo.mutable.load(%buf0, %ch2) : (!mo.buffer<[4, 4], f32>, !mo.chain) -> (!mo.tensor<[4, 4], f32>, !mo.chain)
  %2 = mo.matmul(%loaded_tensor2, %arg2) : (!mo.tensor<[4, 4], f32>, !mo.tensor<[4, 4], f32>) -> !mo.tensor<[4, 4], f32>
  %ch4 = mo.mutable.store(%buf0, %2, %ch3) : (!mo.buffer<[4, 4], f32>, !mo.tensor<[4, 4], f32>, !mo.chain) -> (!mo.chain)

  mo.output %buf0 : !mo.buffer<[4, 4], f32>
}

// CHECK-LABEL: mo.graph @write_layer_norm_store_same_buffer
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[outTensor:%.*]], [[outChain:%.*]] = mo.mutable.load(%arg0, [[ch0]])
// CHECK: [[ch1:%.*]] = mogg.kernel{{.*}}chain([[outChain]])
// CHECK: summary = "mo.layer_norm:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
mo.graph @write_layer_norm_store_same_buffer(%buf: !mo.buffer<[3, 2], f32>, %gamma: !mo.tensor<[2], f32>, %beta: !mo.tensor<[2], f32>, %eps: !mo.tensor<[], f32>) -> !mo.buffer<[3, 2], f32>
{
  %ch0 = mo.chain.create()

  %loaded_tensor, %ch1 = mo.mutable.load(%buf, %ch0) : (!mo.buffer<[3, 2], f32>, !mo.chain) -> (!mo.tensor<[3, 2], f32>, !mo.chain)
  %normalized = mo.layer_norm(%loaded_tensor, %gamma, %beta, %eps) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 2], f32>
  %ch2 = mo.mutable.store(%buf, %normalized, %ch1) : (!mo.buffer<[3, 2], f32>, !mo.tensor<[3, 2], f32>, !mo.chain) -> !mo.chain

  mo.output %buf : !mo.buffer<[3, 2], f32>
}

// CHECK-LABEL: mo.graph @write_layer_norm_store_different_buffers
// CHECK: [[ch0:%.*]] = mo.chain.create()
// CHECK: [[ch1:%.*]] = mo.chain.create([[ch2:%.*]], [[ch0]])
// CHECK: [[ch2]] = mogg.kernel{{.*}}chain([[ch1]])
// CHECK: summary = "mo.mutable.load:{{[0-9]+}}+mo.layer_norm:{{[0-9]+}}+mo.mutable.store:{{[0-9]+}}"
mo.graph @write_layer_norm_store_different_buffers(%src_buf: !mo.buffer<[3, 2], f32>, %dst_buf: !mo.buffer<[3, 2], f32>, %gamma: !mo.tensor<[2], f32>, %beta: !mo.tensor<[2], f32>, %eps: !mo.tensor<[], f32>) -> !mo.buffer<[3, 2], f32>
{
  %ch0 = mo.chain.create()

  %loaded_tensor, %ch1 = mo.mutable.load(%src_buf, %ch0) : (!mo.buffer<[3, 2], f32>, !mo.chain) -> (!mo.tensor<[3, 2], f32>, !mo.chain)

  %normalized = mo.layer_norm(%loaded_tensor, %gamma, %beta, %eps) : (!mo.tensor<[3, 2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[2], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[3, 2], f32>

  %ch2 = mo.mutable.store(%dst_buf, %normalized, %ch1) : (!mo.buffer<[3, 2], f32>, !mo.tensor<[3, 2], f32>, !mo.chain) -> !mo.chain

  mo.output %dst_buf : !mo.buffer<[3, 2], f32>
}
