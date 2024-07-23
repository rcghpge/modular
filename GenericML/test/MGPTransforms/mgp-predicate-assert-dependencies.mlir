// RUN: mo-opt %s --allow-unregistered-dialect --predicate-asserts-deps --simplify-chains --cse | FileCheck %s
//

// CHECK-LABEL: mgp.model @simple_case : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "input shapes must match"
// CHECK: %[[CH3:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH1]], %[[CH2]]: !mef.chain)
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH3]], %[[CTX]][1], %{{.*}}, align 16 : !mgp.buffer<>
// CHECK: %{{.*}} = mgp.kernel.execute %[[CH3]], %[[CTX]][1, 1], (0, 0), (in %{{.*}} : !mgp.buffer<>, in %{{.*}} : !mgp.buffer<>, out %[[BUF]] : !mgp.buffer<>), (%{{.*}}, %{{.*}}),  summary "foo"

mgp.model @simple_case : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch1 = mgp.chain.host_to_device  %ch0, %ctx[1]
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %0:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %1:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %0#0, %0#1 : !mgp.tensor_spec<?x?xf32>
  %7 = mip.nary.mul(%0#0, %0#1) {constInt = 8 : si64}
  %8 = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %7, align 16 : !mgp.buffer<>
  %9 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %bufferResult_2 : !mgp.buffer<>, in %bufferResult_4 : !mgp.buffer<>, out %8 : !mgp.buffer<>), (%0#0, %0#1), summary "foo"
  %10 = mgp.tensor.create %9, %ctx[1], %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL: mgp.model @two_asserts : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "input shapes must match"
// CHECK: %[[CH3:.*]] = mgp.assert %{{.*}}, "input dimension 0 must match"
// CHECK: %[[CH4:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH1]], %[[CH2]]: !mef.chain, %[[CH3]]: !mef.chain)
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH4]], %[[CTX]][1], %{{.*}}, align 16 : !mgp.buffer<>
// CHECK: %{{.*}} = mgp.kernel.execute %[[CH4]], %[[CTX]][1, 1], (0, 0), (in %{{.*}} : !mgp.buffer<>, in %{{.*}} : !mgp.buffer<>, out %[[BUF]] : !mgp.buffer<>), (%2{{.*}}, %{{.*}}),  summary "foo"

mgp.model @two_asserts : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch1 = mgp.chain.host_to_device  %ch0, %ctx[1]
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %0:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %1:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  %added_assert = mgp.assert %2, "input dimension 0 must match"
  %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %0#0, %0#1 : !mgp.tensor_spec<?x?xf32>
  %7 = mip.nary.mul(%0#0, %0#1) {constInt = 8 : si64}
  %8 = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %7, align 16 : !mgp.buffer<>
  %9 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %bufferResult_2 : !mgp.buffer<>, in %bufferResult_4 : !mgp.buffer<>, out %8 : !mgp.buffer<>), (%0#0, %0#1), summary "foo"
  %10 = mgp.tensor.create %9, %ctx[1], %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL: mgp.model @assert_depends_on_control_flow : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "input shapes must match"
// CHECK: %[[CH3:.*]] = mgp.chain.create %[[CTX]][1], (%[[CH1]], %[[CH2]]: !mef.chain)
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH3]], %[[CTX]][1], %{{.*}}, align 16 : !mgp.buffer<>
// CHECK: %{{.*}} = mgp.kernel.execute %[[CH3]], %[[CTX]][1, 1], (0, 0), (in %{{.*}} : !mgp.buffer<>, in %{{.*}} : !mgp.buffer<>, out %[[BUF]] : !mgp.buffer<>), (%{{.*}}, %{{.*}}),  summary "foo"

mgp.model @assert_depends_on_control_flow : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch1 = mgp.chain.host_to_device  %ch0, %ctx[1]
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %0:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %1:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %cf:2 = grt.if %4 (%ch0: !mef.chain, %2: i1, %3: i1) -> (!mef.chain, i1) {
    mef.output %ch0, %2 : !mef.chain, i1
  } else {
    mef.output %ch0, %3 : !mef.chain, i1
  }
  %5 = mgp.assert %cf#1, "input shapes must match"
  %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %0#0, %0#1 : !mgp.tensor_spec<?x?xf32>
  %7 = mip.nary.mul(%0#0, %0#1) {constInt = 8 : si64}
  %8 = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %7, align 16 : !mgp.buffer<>
  %9 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %bufferResult_2 : !mgp.buffer<>, in %bufferResult_4 : !mgp.buffer<>, out %8 : !mgp.buffer<>), (%0#0, %0#1), summary "foo"
  %10 = mgp.tensor.create %9, %ctx[1], %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// NOTE: Currently the assert is ignored since nested blocks within control flow
//       ops are ignored!
// CHECK-LABEL: mgp.model @assert_in_control_flow : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %[[CH1:.*]] = mgp.chain.host_to_device %[[CH0]], %[[CTX]][1]
// CHECK: %[[CH2:.*]] = grt.if %{{.*}} (%{{.*}}: i1, %{{.*}}: i1, %{{.*}}: !mef.chain) -> !mef.chain {
// CHECK:   %[[CH3:.*]] = mgp.assert %{{.*}}, "dimensions must match"
// CHECK:   mef.output %[[CH3]] : !mef.chain
// CHECK: } else {
// CHECK:   mef.output %{{.*}} : !mef.chain
// CHECK: }
// CHECK: %[[BUF:.*]] = mgp.buffer.alloc.dynamic %[[CH1]], %[[CTX]][1], %{{.*}}, align 16 : !mgp.buffer<>
// CHECK: %{{.*}} = mgp.kernel.execute %[[CH1]], %[[CTX]][1, 1], (0, 0), (in %{{.*}} : !mgp.buffer<>, in %{{.*}} : !mgp.buffer<>, out %[[BUF]] : !mgp.buffer<>), (%2#0, %2#1),  summary "foo"

mgp.model @assert_in_control_flow : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>) {
  %ch1 = mgp.chain.host_to_device  %ch0, %ctx[1]
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %0:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %1:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %cf = grt.if %4 (%2: i1, %3: i1, %ch0: !mef.chain) -> (!mef.chain) {
    %5 = mgp.assert %2, "dimensions must match"
    mef.output %5 : !mef.chain
  } else {
    mef.output %ch0 : !mef.chain
  }
  %bufferResult_2 = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %bufferResult_4 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %6 = mgp.tensor_spec.create %0#0, %0#1 : !mgp.tensor_spec<?x?xf32>
  %7 = mip.nary.mul(%0#0, %0#1) {constInt = 8 : si64}
  %8 = mgp.buffer.alloc.dynamic %ch1, %ctx[1], %7, align 16 : !mgp.buffer<>
  %9 = mgp.kernel.execute %ch1, %ctx[1, 1], (0, 0), (in %bufferResult_2 : !mgp.buffer<>, in %bufferResult_4 : !mgp.buffer<>, out %8 : !mgp.buffer<>), (%0#0, %0#1), summary "foo"
  %10 = mgp.tensor.create %9, %ctx[1], %8, %6  : !mgp.tensor<?x?xf32>
  mef.output %ch0, %10 : !mef.chain, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL: mgp.model @no_predication : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context,
// CHECK: %{{.*}} = mgp.assert %{{.*}}, "input shapes must match"
// CHECK-NEXT: mef.output %[[CH0]] : !mef.chain

mgp.model @no_predication : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain) {
  %ch1 = mgp.chain.host_to_device  %ch0, %ctx[1]
  %bufferResult = mgp.tensor.extract.buffer %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %tensorSpecResult = mgp.tensor.extract.tensor_spec %ctx[1], %arg2 : !mgp.tensor<?x?xf32>
  %0:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.tensor_spec<?x?xf32>
  %bufferResult_0 = mgp.tensor.extract.buffer %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %tensorSpecResult_1 = mgp.tensor.extract.tensor_spec %ctx[1], %arg3 : !mgp.tensor<?x?xf32>
  %1:2 = mgp.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  mef.output %ch0 : !mef.chain
}

// CHECK-LABEL: mgp.model @on_output_op : mof devices [#M.device_ref<"cpu", 0>] init
// CHECK: execute (%[[CH0:.*]]: !mef.chain,
// CHECK: %[[CH1:.*]] = mgp.assert %{{.*}}, "Tensor type mismatch for argument 0"
// CHECK: %[[CH2:.*]] = mgp.assert %{{.*}}, "Tensor type mismatch for argument 1"
// CHECK: %[[CH3:.*]] = grt.chain.create (%[[CH0]], %[[CH1]], %[[CH2]])
// CHECK: mef.output %[[CH3]]

mgp.model @on_output_op : mof devices [#M.device_ref<"cpu", 0>] init (%arg0: !mgp.opaque) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%ch0: !mef.chain, %ctx: !mgp.context, %cpu: !mgp.opaque, %arg2: !mgp.tensor<2x4xf32>, %arg3: !mgp.tensor<2x4xf32>) -> (!mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>) {
  %0 = mgp.tensor.extract.tensor_spec %ctx[1], %arg2 shape_only true : !mgp.tensor<2x4xf32>
  %1 = mgp.tensor_spec.equal.static %0 : !mgp.tensor_spec<2x4xunk>
  %2 = mgp.assert %1, "Tensor type mismatch for argument 0"
  %3 = mgp.tensor.extract.tensor_spec %ctx[1], %arg3 shape_only true : !mgp.tensor<2x4xf32>
  %4 = mgp.tensor_spec.equal.static %3 : !mgp.tensor_spec<2x4xunk>
  %5 = mgp.assert %4, "Tensor type mismatch for argument 1"
  mef.output %ch0, %arg2, %arg3 : !mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>
}
