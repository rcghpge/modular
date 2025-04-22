// RUN: mo-opt %s --mo-to-mogg | FileCheck %s
// TODO(GEX-1573): Convert the following sniff test to a complete test.
// RUN: MOGG_USE_EXP_KERNELS=1 mo-opt %s --mo-to-mogg | FileCheck %s --check-prefix="CHECK-MOGG"

mo.graph @bcast_basic<N, L, M, K>(%arg0: !mo.tensor<[N, M, L], f32>) -> (!mo.tensor<[N, M, K], f32>) {
  %0 = mo.static.broadcast_to(%arg0) : !mo.tensor<[N, M, L], f32> -> !mo.tensor<[N, M, K], f32>
  mo.output %0: !mo.tensor<[N, M, K], f32>
}

// CHECK-LABEL: mo.graph @bcast_basic
// CHECK: %[[KERNEL:.*]] = mogg.kernel (%[[NULL:.*]]) : (%[[IN1:.*]]: !mo.tensor<[N, M, L], f32>) -> !mo.tensor<[N, M, K], f32>
// CHECK: %[[OUT:.*]] = mogg.output_placeholders : !mo.tensor<[N, M, K], f32>

// CHECK-DAG: %[[N:.*]] = mogg.get_dim <N>
// CHECK-DAG: %[[M:.*]] = mogg.get_dim <M>
// CHECK-DAG: %[[K:.*]] = mogg.get_dim <K>
// CHECK-DAG: %[[ARR:.*]] = pop.array.create [%[[N]], %[[M]], %[[K]]]

// CHECK: kgen.call @"MOGGKernelAPILib::MOGGKernelAPI::StaticBroadcastTo::update_input_view
// CHECK: kgen.call @"tensor_internal::managed_tensor_slice::view_copy_impl

// CHECK: mogg.output %[[OUT]] : !mo.tensor<[N, M, K], f32>
// CHECK: device = #M.device_ref<"cpu", 0>, inputFusionInterface = [""], isAsync = true, isViewOp = true
// CHECK: mo.output %[[KERNEL]] : !mo.tensor<[N, M, K], f32>


// CHECK-MOGG-LABEL: mo.graph @bcast_basic
// CHECK-MOGG:         mogg.call.update_view["mo.static.broadcast_to"]
// CHECK-MOGG:         mogg.call.materialize
