// TODO(GEX-2735): testfuse ops don't implement EmitsMojoInterface
// XFAIL: *

// RUN: mo-opt --allow-unregistered-dialect --mgp-lower-primitives --mogg-sequence-fusion --mogg-remove-redundant-operands --mogg-outline-kernels --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-jit-compile-kernels --mogg-lower-mogg-region-compile %s -o %t.mlir
// RUN: mt --execute %t.mlir | FileCheck --check-prefix=FUSE-CHECK %s
// RUN: mo-opt --allow-unregistered-dialect --mgp-lower-primitives --mogg-remove-redundant-operands --mogg-outline-kernels --mogg-add-try-except-blocks --canonicalize --mogg-emit-mojo --mogg-jit-compile-kernels --mogg-lower-mogg-region-compile %s -o %t_no_fused.mlir
// RUN: mt --execute %t_no_fused.mlir | FileCheck %s

// FUSE-CHECK-LABEL: --- Running 'test_myint':
// FUSE-CHECK-NEXT: MyInt.__del__ 8
// FUSE-CHECK-NEXT: MyInt.__del__ 64
// CHECK-NOT:  <error: Assert>

// CHECK-LABEL: --- Running 'test_myint':
// CHECK-NEXT: MyInt.__moveinit__ 8
// CHECK-NEXT: MyInt.__moveinit__ 64
// CHECK-NEXT: MyInt.__del__ 8
// CHECK-NEXT: MyInt.__del__ 64
// CHECK-NOT:  <error: Assert>

mgp.model @test_myint: mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %idx = index.constant 8
  %0 = "testfuse.my_int.from_index"(%idx) : (index) -> !mgp.opaque<"">
  %1 = "testfuse.my_int.square"(%0) : (!mgp.opaque<"">) -> !mgp.opaque<"">
  %res = "testfuse.my_int.to_index"(%1) : (!mgp.opaque<"">) -> index
  %fact = index.constant 64
  %cond = index.cmp eq(%res, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}

// FUSE-CHECK-LABEL: --- Running 'test_myintreg2':
// FUSE-CHECK-NEXT: MyIntReg2.__del__ 7
// FUSE-CHECK-NEXT: MyIntReg2.__del__ 49
// CHECK-NOT:  <error: Assert>

// CHECK-LABEL: --- Running 'test_myintreg2':
// CHECK-NEXT: MyIntReg2.__del__ 7
// CHECK-NEXT: MyIntReg2.__del__ 49
// CHECK-NOT:  <error: Assert>

mgp.model @test_myintreg2: mof devices [#M.device_ref<"cpu", 0>] init (%weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch2 = mgp.runtime.create %ch0, %ctx[1]
  mef.output %ch2, %ctx, %weights, %cpu : !mef.chain, !mgp.context, !mgp.opaque<"weights_registry">, !mgp.device_context<<"cpu", 0>>
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %weights: !mgp.opaque<"weights_registry">, %cpu: !mgp.device_context<<"cpu", 0>>) -> !mef.chain {
  %ch0 = grt.chain.create ()
  %idx = index.constant 7
  %0 = "testfuse.my_int_reg2.from_index"(%idx) : (index) -> !mgp.opaque<"">
  %1 = "testfuse.my_int_reg2.square"(%0) : (!mgp.opaque<"">) -> !mgp.opaque<"">
  %res = "testfuse.my_int_reg2.to_index"(%1) : (!mgp.opaque<"">) -> index
  %fact = index.constant 49
  %cond = index.cmp eq(%res, %fact)
  %ch1 = mgp.assert %cond, "ASSERT"
  mef.output %ch1 : !mef.chain
}
