// RUN: mo-opt --canonicalize --cse %s | FileCheck %s

// The kernel fills a buffer its operands point to, so neither an unused
// result nor an identical earlier call makes the call removable.
// CHECK-LABEL: func.func @fill_twice
// CHECK:         mogg.execute_kernel @fill
// CHECK:         mogg.execute_kernel @fill
// CHECK:         return
func.func @fill_twice(%ch: !mef.chain, %buf: !mgp.buffer, %ctx: !mgp.device_context<<"cpu", 0>>) {
  %0 = mogg.execute_kernel @fill(%ch, %buf, %ctx) constantResources = [], isAsync = true, usedParams = [] {device = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  %1 = mogg.execute_kernel @fill(%ch, %buf, %ctx) constantResources = [], isAsync = true, usedParams = [] {device = #M.device_ref<"cpu", 0>} : (!mef.chain, !mgp.buffer, !mgp.device_context<<"cpu", 0>>) -> !mef.chain
  return
}
