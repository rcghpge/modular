// RUN: mo-opt --lower-mogg-region-compile %s | FileCheck %s

// Jit compile kernels will output the function execution in a particular way,
// we need to run a fixup pass over it to convert to the form MGP expects.

mgp.model @basic : mof init {
  "mogg.load_compiled_kernel"() {symbols = ["__gen_0"], value = dense_resource<fake_binary> : tensor<1xui8>} : () -> ()
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %ch1 = mgp.check_runtime %0, %1[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %2 = mgp.runtime.create %ch1, %1[0, 1] device<"cpu", 0>
  mef.output %2, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context) -> !mef.chain {
  %0 = grt.chain.create ()
  %1 = mip.constant.index 42
  %2 = mip.constant.index -20
  %3 = "mogg.execute_kernel"(%1, %2) {hasCuda=false, isAsync = false, kernelName = "__gen_0", outputParamDecls = #kgen<param.decls[]>, summary = "mogg.generic.compiled.region:14", usedParams = []} : (index, index) -> index
  %4 = mip.print.index %3, %0
  mef.output %4 : !mef.chain
}

// CHECK-LABEL: mgp.model @basic
// CHECK: %[[CTX:.*]] = mgp.context.create 2
// CHECK: %[[KERNEL_CTX:.*]] = mgp.runtime.create
// CHECK-NEXT: mgp.module.load %[[KERNEL_CTX]], %[[CTX]][0], dense_resource<fake_binary> : tensor<1xui8>, ["__gen_0"]

// CHECK: } execute
// CHECK: "mgp.generic.execute"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{{.*}}, dSummary = {{.*}}} : (!mgp.context, !mef.chain, index, index) -> (!mgp.chain, index)

mgp.model @basic_control_flow : mof init {
  "mogg.load_compiled_kernel"() {symbols = ["__gen_0"], value = dense_resource<fake_binary> : tensor<1xui8>} : () -> ()
  %0 = grt.chain.create ()
  %1 = mgp.context.create 2
  %ch1 = mgp.check_runtime %0, %1[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %2 = mgp.runtime.create %ch1, %1[0, 1] device<"cpu", 0>
  mef.output %2, %1 : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.buffer<1>) -> !mef.chain {
  %ch0 = mgp.chain.host_to_device %arg0, %arg1[1]
  %lhs = mip.constant.index 42
  %rhs = mip.constant.index -20
  %1 = mgp.buffer.to_bool %ch0, %arg1[1], %arg2 : !mgp.buffer<1>
  %2 = grt.if %1 (%ch0: !mgp.chain, %lhs: index, %rhs: index, %arg1 as %new: !mgp.context) -> !mgp.chain {
    %3 = "mogg.execute_kernel"(%lhs, %rhs) {hasCuda=false, isAsync = false, kernelName = "__gen_0", outputParamDecls = #kgen<param.decls[]>, summary = "mogg.generic.compiled.region:14", usedParams = []} : (index, index) -> index
    mef.output %ch0 : !mgp.chain
  } else {
    mef.output %ch0 : !mgp.chain
  }
  %ch1 = mgp.chain.device_to_host %2, %arg1[1]
  mef.output %ch1 : !mef.chain
}

// CHECK-LABEL: mgp.model @basic_control_flow
// CHECK: %[[CTX:.*]] = mgp.context.create 2
// CHECK: %[[KERNEL_CTX:.*]] = mgp.runtime.create
// CHECK-NEXT: mgp.module.load %[[KERNEL_CTX]], %[[CTX]][0], dense_resource<fake_binary> : tensor<1xui8>, ["__gen_0"]
// CHECK: } execute
// CHECK: "mgp.generic.execute"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{{.*}}, dSummary = {{.*}}} : (!mgp.context, !mef.chain, index, index) -> (!mgp.chain, index)
