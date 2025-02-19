// RUN: mo-opt --mo-to-primitives %s | FileCheck --enable-var-scope %s

// CHECK-LABEL: mgp.model @buffer_transfer_h2d
// CHECK: execute
// CHECK-SAME: (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WREG:.*]]: !mgp.opaque<"weights_registry">, %[[CPU_CTX:.*]]: !mgp.device_context<<"cpu", 0>>, %[[GPU_CTX:.*]]: !mgp.device_context<<"gpu", 0>>, %[[ARG:.*]]: !mgp.tensor<2x2xf32>)
// CHECK: %[[CH0:.*]] = grt.chain.create ()
// CHECK: %[[SPEC:.*]] = mgp.tensor_spec.create
// CHECK: %[[IDX4:.*]] = index.constant 4
// CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]]
// CHECK: %[[MUL0:.*]] = index.mul %[[IDX4]], %[[DIM0]]
// CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]]
// CHECK: %[[MUL1:.*]] = index.mul %[[MUL0]], %[[DIM1]]
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[MUL1]], %[[GPU_CTX]], align 256
// CHECK: %[[TENSOR:.*]] = mgp.tensor.create %[[CHAIN]], %[[BUF]], %[[SPEC]]
// CHECK: %[[SRC_BUF:.*]] = mgp.tensor.extract.buffer %[[ARG]]
// CHECK: %[[SRC_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[ARG]]
// CHECK: %[[DST_BUF:.*]] = mgp.tensor.extract.buffer %[[TENSOR]]
// CHECK: %[[DST_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[TENSOR]]
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[CHAIN]], %[[CH0]])
// CHECK: %[[COPY:.*]] = mgp.buffer.host_to_device %[[CH1]], %[[SRC_BUF]], %[[DST_BUF]], %[[GPU_CTX]]
// CHECK: %[[SYNC:.*]] = mgp.sync(%[[COPY]], %[[CTX]], %[[GPU_CTX]]) {bDevice = #M.device_ref<"gpu", 0>}
// CHECK: %[[EXTEND:.*]] = mgp.extend.lifetime %[[SYNC]], %[[SRC_BUF]]
// CHECK: mef.output %[[CHAIN]], %[[TENSOR]]
mo.graph @buffer_transfer_h2d(%arg0: !mo.buffer<[2, 2], f32, cpu:0>) -> !mo.buffer<[2, 2], f32, gpu:0>  {
  %ch = mo.chain.create()
  %darg0 = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:0>

  mo.buffer.transfer(%ch, %arg0, %darg0) : !mo.buffer<[2, 2], f32, cpu:0>, !mo.buffer<[2, 2], f32, gpu:0>

  mo.output %darg0: !mo.buffer<[2, 2], f32, gpu:0>
}

// CHECK-LABEL: mgp.model @buffer_transfer_d2d
// CHECK: execute
// CHECK-SAME: (%[[CHAIN:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[WREG:.*]]: !mgp.opaque<"weights_registry">, %[[CPU_CTX:.*]]: !mgp.device_context<<"cpu", 0>>, %[[GPU0_CTX:.*]]: !mgp.device_context<<"gpu", 0>>, %[[GPU1_CTX:.*]]: !mgp.device_context<<"gpu", 1>>, %[[ARG:.*]]: !mgp.tensor<2x2xf32, <"gpu", 0>>)
// CHECK: %[[CH0:.*]] = grt.chain.create ()
// CHECK: %[[SPEC:.*]] = mgp.tensor_spec.create
// CHECK: %[[IDX4:.*]] = index.constant 4
// CHECK: %[[DIM0:.*]] = mgp.tensor_spec.get_dim[0] %[[SPEC]]
// CHECK: %[[MUL0:.*]] = index.mul %[[IDX4]], %[[DIM0]]
// CHECK: %[[DIM1:.*]] = mgp.tensor_spec.get_dim[1] %[[SPEC]]
// CHECK: %[[MUL1:.*]] = index.mul %[[MUL0]], %[[DIM1]]
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc %[[CHAIN]], %[[MUL1]], %[[GPU1_CTX]], align 256
// CHECK: %[[TENSOR:.*]] = mgp.tensor.create %[[CHAIN]], %[[BUF]], %[[SPEC]]
// CHECK: %[[SRC_BUF:.*]] = mgp.tensor.extract.buffer %[[ARG]]
// CHECK: %[[SRC_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[ARG]]
// CHECK: %[[DST_BUF:.*]] = mgp.tensor.extract.buffer %[[TENSOR]]
// CHECK: %[[DST_SPEC:.*]] = mgp.tensor.extract.tensor_spec %[[TENSOR]]
// CHECK: %[[CH1:.*]] = grt.chain.create (%[[CHAIN]], %[[CH0]])
// CHECK: %[[COPY:.*]] = mgp.buffer.device_to_device %[[CH1]], %[[SRC_BUF]], %[[GPU0_CTX]], %[[DST_BUF]], %[[GPU1_CTX]]
// CHECK: mef.output %[[CHAIN]], %[[TENSOR]]
mo.graph @buffer_transfer_d2d(%arg0: !mo.buffer<[2, 2], f32, gpu:0>) -> !mo.buffer<[2, 2], f32, gpu:1>  {
  %ch = mo.chain.create()
  %darg0 = mo.buffer.create : !mo.buffer<[2, 2], f32, gpu:1>

  mo.buffer.transfer(%ch, %arg0, %darg0) : !mo.buffer<[2, 2], f32, gpu:0>, !mo.buffer<[2, 2], f32, gpu:1>

  mo.output %darg0: !mo.buffer<[2, 2], f32, gpu:1>
}
