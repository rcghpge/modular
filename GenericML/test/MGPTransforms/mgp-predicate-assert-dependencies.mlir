// RUN: mgp-opt %s --allow-unregistered-dialect --predicate-asserts-deps --simplify-chains | FileCheck %s

// CHECK-LABEL: @simple_case
// CHECK-NEXT:   %0 = grt.chain.create ()
// CHECK-NEXT:   %1 = mgp.context.create 1
// CHECK-NEXT:   mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// CHECK-NEXT:   %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %2 = mip.cmp.eq %0#0, %1#0
// CHECK-NEXT:   %3 = mip.cmp.eq %0#1, %1#1
// CHECK-NEXT:   %4 = mip.bool.and %2, %3
// CHECK-NEXT:   %5 = mgp.assert %4, "input shapes must match"
// CHECK-NEXT:   %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %6 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %7 = mgp.cpu.buffer.alloc.spec %5, %6, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %8 = "mop.kgen.execute_kernel"(%5, %arg1, %bufferResult_2, %bufferResult_4, %7, %0#0, %0#1) : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
// CHECK-NEXT:   %9 = mgp.cpu.tensor.create %8, %7, %6  : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   mef.output %arg0, %9 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT: }

mgp.model @simple_case : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
  %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %6 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
  %7 = mgp.cpu.buffer.alloc.spec %arg0, %6, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %8 = "mop.kgen.execute_kernel"(%arg0, %arg1, %bufferResult_2, %bufferResult_4, %7, %0#0, %0#1) {} : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
  %9 = mgp.cpu.tensor.create %8, %7, %6  : !mgp.cpu.tensor<?x?xf32>
  mef.output %arg0, %9 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
}

// CHECK-LABEL: @two_asserts
// CHECK-NEXT:   %0 = grt.chain.create ()
// CHECK-NEXT:   %1 = mgp.context.create 1
// CHECK-NEXT:   mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// CHECK-NEXT:   %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %2 = mip.cmp.eq %0#0, %1#0
// CHECK-NEXT:   %3 = mip.cmp.eq %0#1, %1#1
// CHECK-NEXT:   %4 = mip.bool.and %2, %3
// CHECK-NEXT:   %5 = mgp.assert %4, "input shapes must match"
// CHECK-NEXT:   %6 = mgp.assert %2, "input dimension 0 must match"
// CHECK-NEXT:   %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %7 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %8 = grt.chain.create ({{%5, %6|%6, %5}})
// CHECK-NEXT:   %9 = mgp.cpu.buffer.alloc.spec %8, %7, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %10 = grt.chain.create ({{%5, %6|%6, %5}})
// CHECK-NEXT:   %11 = "mop.kgen.execute_kernel"(%10, %arg1, %bufferResult_2, %bufferResult_4, %9, %0#0, %0#1) : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
// CHECK-NEXT:   %12 = mgp.cpu.tensor.create %11, %9, %7  : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   mef.output %arg0, %12 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT: }

mgp.model @two_asserts : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
  %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  %added_assert = mgp.assert %2, "input dimension 0 must match"
  %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %6 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
  %7 = mgp.cpu.buffer.alloc.spec %arg0, %6, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %8 = "mop.kgen.execute_kernel"(%arg0, %arg1, %bufferResult_2, %bufferResult_4, %7, %0#0, %0#1) {} : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
  %9 = mgp.cpu.tensor.create %8, %7, %6  : !mgp.cpu.tensor<?x?xf32>
  mef.output %arg0, %9 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
}

// CHECK-LABEL: @assert_depends_on_control_flow
// CHECK-NEXT:   %0 = grt.chain.create ()
// CHECK-NEXT:   %1 = mgp.context.create 1
// CHECK-NEXT:   mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// CHECK-NEXT:   %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %2 = mip.cmp.eq %0#0, %1#0
// CHECK-NEXT:   %3 = mip.cmp.eq %0#1, %1#1
// CHECK-NEXT:   %4 = mip.bool.and %2, %3
// CHECK-NEXT:   %5:2 = grt.if %4 (%arg0: !mef.chain, %2: i1, %3: i1) -> (!mef.chain, i1) {
// CHECK-NEXT:     mef.output %arg0, %2 : !mef.chain, i1
// CHECK-NEXT:   } else {
// CHECK-NEXT:     mef.output %arg0, %3 : !mef.chain, i1
// CHECK-NEXT:   }
// CHECK-NEXT:   %6 = mgp.assert %5#1, "input shapes must match"
// CHECK-NEXT:   %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %7 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %8 = mgp.cpu.buffer.alloc.spec %6, %7, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %9 = "mop.kgen.execute_kernel"(%6, %arg1, %bufferResult_2, %bufferResult_4, %8, %0#0, %0#1) : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
// CHECK-NEXT:   %10 = mgp.cpu.tensor.create %9, %8, %7  : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   mef.output %6, %10 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT: }

mgp.model @assert_depends_on_control_flow : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
  %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %cf:2 = grt.if %4 (%arg0: !mef.chain, %2: i1, %3: i1) -> (!mef.chain, i1) {
    mef.output %arg0, %2 : !mef.chain, i1
  } else {
    mef.output %arg0, %3 : !mef.chain, i1
  }
  %5 = mgp.assert %cf#1, "input shapes must match"
  %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %6 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
  %7 = mgp.cpu.buffer.alloc.spec %arg0, %6, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %8 = "mop.kgen.execute_kernel"(%arg0, %arg1, %bufferResult_2, %bufferResult_4, %7, %0#0, %0#1) {} : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
  %9 = mgp.cpu.tensor.create %8, %7, %6  : !mgp.cpu.tensor<?x?xf32>
  mef.output %arg0, %9 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
}

// CHECK-LABEL: @assert_in_control_flow
// CHECK-NEXT:   %0 = grt.chain.create ()
// CHECK-NEXT:   %1 = mgp.context.create 1
// CHECK-NEXT:   mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
// CHECK-NEXT:   %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %2 = mip.cmp.eq %0#0, %1#0
// CHECK-NEXT:   %3 = mip.cmp.eq %0#1, %1#1
// CHECK-NEXT:   %4 = mip.bool.and %2, %3
// CHECK-NEXT:   %5 = grt.if %4 (%2: i1, %3: i1, %arg0: !mef.chain) -> !mef.chain {
// CHECK-NEXT:     %10 = mgp.assert %2, "dimensions must match"
// CHECK-NEXT:     mef.output %10 : !mef.chain
// CHECK-NEXT:   } else {
// CHECK-NEXT:     mef.output %arg0 : !mef.chain
// CHECK-NEXT:   }
// CHECK-NEXT:   %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %6 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %7 = mgp.cpu.buffer.alloc.spec %arg0, %6, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %8 = "mop.kgen.execute_kernel"(%arg0, %arg1, %bufferResult_2, %bufferResult_4, %7, %0#0, %0#1) : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
// CHECK-NEXT:   %9 = mgp.cpu.tensor.create %8, %7, %6  : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   mef.output %arg0, %9 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT: }

mgp.model @assert_in_control_flow : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>) {
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
  %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %cf = grt.if %4 (%2: i1, %3: i1, %arg0: !mef.chain) -> (!mef.chain) {
    %5 = mgp.assert %2, "dimensions must match"
    mef.output %5 : !mef.chain
  } else {
    mef.output %arg0 : !mef.chain
  }
  %bufferResult_2, %tensorSpecResult_3 = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %bufferResult_4, %tensorSpecResult_5 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %6 = mgp.cpu.tensor_spec.create %0#0, %0#1 : !mgp.cpu.tensor_spec<?x?xf32>
  %7 = mgp.cpu.buffer.alloc.spec %arg0, %6, align 16 : !mgp.cpu.tensor_spec<?x?xf32>
  %8 = "mop.kgen.execute_kernel"(%arg0, %arg1, %bufferResult_2, %bufferResult_4, %7, %0#0, %0#1) {} : (!mef.chain, !mgp.context, !mgp.cpu.buffer, !mgp.cpu.buffer, !mgp.cpu.buffer, index, index) -> !mef.chain
  %9 = mgp.cpu.tensor.create %8, %7, %6  : !mgp.cpu.tensor<?x?xf32>
  mef.output %arg0, %9 : !mef.chain, !mgp.cpu.tensor<?x?xf32>
}

// CHECK-LABEL: @no_predication
// CHECK-NEXT:   %0 = grt.chain.create ()
// CHECK-NEXT:   %1 = mgp.context.create 1
// CHECK-NEXT:   mef.output %0, %1 : !mef.chain, !mgp.context
// CHECK-NEXT: } execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> !mef.chain {
// CHECK-NEXT:   %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
// CHECK-NEXT:   %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
// CHECK-NEXT:   %2 = mip.cmp.eq %0#0, %1#0
// CHECK-NEXT:   %3 = mip.cmp.eq %0#1, %1#1
// CHECK-NEXT:   %4 = mip.bool.and %2, %3
// CHECK-NEXT:   %5 = mgp.assert %4, "input shapes must match"
// CHECK-NEXT:   mef.output %arg0 : !mef.chain
// CHECK-NEXT: }

mgp.model @no_predication : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain) {
  %bufferResult, %tensorSpecResult = mgp.cpu.tensor.extract %arg2 : !mgp.cpu.tensor<?x?xf32>
  %0:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult : !mgp.cpu.tensor_spec<?x?xf32>
  %bufferResult_0, %tensorSpecResult_1 = mgp.cpu.tensor.extract %arg3 : !mgp.cpu.tensor<?x?xf32>
  %1:2 = mgp.cpu.tensor_spec.dims.dynamic %tensorSpecResult_1 : !mgp.cpu.tensor_spec<?x?xf32>
  %2 = mip.cmp.eq %0#0, %1#0
  %3 = mip.cmp.eq %0#1, %1#1
  %4 = mip.bool.and %2, %3
  %5 = mgp.assert %4, "input shapes must match"
  mef.output %arg0 : !mef.chain
}
