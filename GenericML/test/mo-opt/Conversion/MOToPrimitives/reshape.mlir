// RUN: mo-opt %s --mgp-mo-to-primitives | FileCheck %s

// Entry point
// CHECK-LABEL:   mgp.model @static_reshape : mof
//
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK-SAME: %[[ARG2:[^:]*]]: !mgp.tensor<?x?xf32>, %[[ARG3:.*]]: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {

// CHECK:   %[[SPEC3:.*]] = mgp.tensor.extract.tensor_spec %[[ARG2]]
// CHECK:   %[[DIM0:.:*]] = mgp.tensor_spec.get_dim[0] %[[SPEC3]]
// CHECK:   %[[DIM1:.:*]] = mgp.tensor_spec.get_dim[1] %[[SPEC3]]

// CHECK:   %[[SPEC1:.*]] = mgp.tensor.extract.tensor_spec %[[ARG3]]
// CHECK:   %[[RESIZED_DIM0:.:*]] = mgp.tensor_spec.get_dim[0] %[[SPEC1]]
// CHECK:   %[[RESIZED_DIM1:.:*]] = mgp.tensor_spec.get_dim[1] %[[SPEC1]]

// CHECK:   %[[CASTED0:.*]] = index.casts %[[RESIZED_DIM0]] : index to si64
// CHECK:   %[[CASTED1:.*]] = index.casts %[[RESIZED_DIM1]] : index to si64

// CHECK:   %[[BUF:.*]] = mgp.tensor.extract.buffer %[[ARG2]]
// CHECK:   %[[CASTED_BACK0:.*]] = index.casts %[[CASTED0]] : si64 to index
// CHECK:   %[[CASTED_BACK1:.*]] = index.casts %[[CASTED1]] : si64 to index

// CHECK:   %[[NEW_SPEC:.*]] = mgp.tensor_spec.create %[[CASTED_BACK0]], %[[CASTED_BACK1]] : !mgp.tensor_spec<?x?xf32>

// CHECK:   %[[OUT:.*]] = mgp.tensor.create %[[CH0]], %[[BUF]], %[[NEW_SPEC]] : !mgp.tensor<?x?xf32>
// CHECK:   mef.output %[[CH0]], %[[OUT]]

mo.graph @static_reshape<D4, D5, D6, D7>(%arg0: !mo.tensor<[D4, D5], f32>, %arg1: !mo.tensor<[D6, D7], f32>) -> !mo.tensor<[D6, D7], f32> {
  %1 = mo.static.reshape(%arg0) : !mo.tensor<[D4, D5], f32> -> !mo.tensor<[D6, D7], f32>
  mo.output %1 : !mo.tensor<[D6, D7], f32>
}

// CHECK-LABEL: mgp.model @multi_reshape : mof
// CHECK: } execute (%[[CH0:.*]]: !mef.chain,

// CHECK: mgp.tensor.create %[[CH0]]

// CHECK: %[[CH1:.*]] = grt.chain.create (%[[CH0]], %[[CH0]])
// CHECK: mgp.tensor.create %[[CH1]]

// CHECK: %[[CH2:.*]] = grt.chain.create (%[[CH1]], %[[CH0]])
// CHECK: mgp.tensor.create %[[CH2]]

mo.graph @multi_reshape<D4, D5, D6, D7>(%arg0: !mo.tensor<[D4, D5], f32>, %arg1: !mo.tensor<[D6, D7], f32>) -> !mo.tensor<[D7, D7], f32> {
  %1 = mo.static.reshape(%arg0) : !mo.tensor<[D4, D5], f32> -> !mo.tensor<[D6, D7], f32>
  %2 = mo.static.reshape(%1) : !mo.tensor<[D6, D7], f32> -> !mo.tensor<[D5, D4], f32>
  %3 = mo.static.reshape(%2) : !mo.tensor<[D5, D4], f32> -> !mo.tensor<[D7, D7], f32>
  mo.output %3 : !mo.tensor<[D7, D7], f32>
}

// CHECK-LABEL: mgp.model @reshape_singleton_dims
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, {{.*}}, %[[ARG3:.*]]: !mgp.tensor<si32>)

// CHECK: %[[IN_BUF:.*]] = mgp.tensor.extract.buffer %[[ARG3]] : <si32>

// CHECK: %[[SPEC:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<1xsi32>
// CHECK: %[[T:.*]] = mgp.tensor.create %[[CH0]], %[[IN_BUF]], %[[SPEC]] : !mgp.tensor<1xsi32>

// CHECK: %[[SPEC2:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<1x1xsi32>

// CHECK: %[[CH:.*]] = grt.chain.create (%[[CH0]], %[[CH0]])
// CHECK: mgp.tensor.create %[[CH]], %[[IN_BUF]], %[[SPEC2]] : !mgp.tensor<1x1xsi32>

mo.graph @reshape_singleton_dims(%in1: !mo.tensor<[], si32>) -> (!mo.tensor<[1, 1], si32>) {
  %1 = mo.add_singleton_dim[0] (%in1) : (!mo.tensor<[], si32>) -> !mo.tensor<[1], si32>
  %2 = mo.add_singleton_dim[1] (%1) : (!mo.tensor<[1], si32>) -> !mo.tensor<[1, 1], si32>
  mo.output %2 : !mo.tensor<[1, 1], si32>
}

// CHECK-LABEL: mgp.model @split_dim_mgp
// CHECK: } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context

// CHECK: %[[BUF:.*]] = mgp.tensor.extract.buffer
// CHECK: %[[SPEC:.*]] = mgp.tensor_spec.create  : !mgp.tensor_spec<10x5xsi32>
// CHECK: mgp.tensor.create %[[CH0]], %[[BUF]], %[[SPEC]] : !mgp.tensor<10x5xsi32>

mo.graph @split_dim_mgp<N>(%arg0: !mo.tensor<[N], si32>) -> !mo.tensor<[10, 5], si32> {
  %0 = mo.split_dim[0] (%arg0) : (!mo.tensor<[N], si32>) -> !mo.tensor<[10, 5], si32>
  mo.output %0 : !mo.tensor<[10, 5], si32>
}
