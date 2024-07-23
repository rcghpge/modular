// RUN: mo-opt %s --add-devices --mo-to-mogg --translate-mogg-to-kgen --inline-lambdas --mo-to-primitives --lower-primitives --lower-abi --canonicalize -cse --jit-compile-kernels --lower-mogg-region-compile | FileCheck %s
// CHECK-LABEL: mgp.model
// CHECK-SAME: @while_op : mof devices [#M.device_ref<"cpu", 0>] init ([[VARG0:[^ ]*]]: !mgp.opaque) {
// CHECK-NEXT: [[V0:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V1:[^ ]*]] = mgp.context.create 1
// CHECK-NEXT: [[V3:[^ ]*]] = mgp.runtime.create [[V0]], [[V1]][0]
// CHECK-NEXT: [[V4:[^ ]*]] = mgp.module.load [[V3]], [[V1]][0] [0],
// CHECK-NEXT: [[V5:[^ ]*]] = grt.chain.create ([[V4]], [[V1]]: !mgp.context)
// CHECK-NEXT: [[V6:[^ ]*]] = mgp.device.context.extract [[V5]], [[V1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN:[^ ]*]], [[VOUTRESULTS:[^ ]*]] = "mgp.generic.execute"([[V1]], [[V3]], [[V6]], [[V4]], [[V1]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 0 : ui64, dSummary = "mgp.device.context.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque, !mef.chain, !mgp.context) -> (!mgp.chain, !mef.chain)
// CHECK-NEXT: mef.output [[VOUTRESULTS]], [[V1]] : !mef.chain, !mgp.context
// CHECK-NEXT: } execute ([[VARG0]]: !mef.chain, [[VARG1:[^ ]*]]: !mgp.context, [[VARG2:[^ ]*]]: !mgp.opaque) -> (!mef.chain, !mgp.tensor<2xf32>) {
// CHECK-NEXT: [[V0]] = mgp.buffer.constant [[VARG0]], [[VARG1]][0], #M.aligned_bytes<"0x0000000000000000"
// CHECK-NEXT: [[V1]] = grt.chain.create ()
// CHECK-NEXT: [[V2:[^ ]*]] = mgp.device.context.extract [[V1]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN]], [[VOUTRESULTS]] = "mgp.generic.execute"([[VARG1]], [[V1]], [[V2]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 1 : ui64, dSummary = "mgp.tensor_spec.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque) -> (!mgp.chain, !mgp.tensor_spec<si64>)
// CHECK-NEXT: [[V3:[^ ]*]] = mgp.chain.host_to_device [[VARG0]], [[VARG1]][0]
// CHECK-NEXT: [[V4:[^ ]*]] = mgp.tensor.create [[V3]], [[VARG1]][0], [[V0]], [[VOUTRESULTS]] : !mgp.tensor<si64>
// CHECK-NEXT: [[V5:[^ ]*]] = mgp.buffer.constant [[VARG0]], [[VARG1]][0], #M.aligned_bytes<"0x0000803F0000803F"
// CHECK-NEXT: [[V6:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V7:[^ ]*]] = mgp.device.context.extract [[V6]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_0:[^ ]*]], [[VOUTRESULTS_1:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V6]], [[V7]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 2 : ui64, dSummary = "mgp.tensor_spec.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque) -> (!mgp.chain, !mgp.tensor_spec<2xf32>)
// CHECK-NEXT: [[V8:[^ ]*]] = mgp.tensor.create [[V3]], [[VARG1]][0], [[V5]], [[VOUTRESULTS_1]] : !mgp.tensor<2xf32>
// CHECK-NEXT: [[V9:[^ ]*]] = mgp.buffer.constant [[VARG0]], [[VARG1]][0], #M.aligned_bytes<"0x00008040"
// CHECK-NEXT: [[V10:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V11:[^ ]*]] = mgp.device.context.extract [[V10]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_2:[^ ]*]], [[VOUTRESULTS_3:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V10]], [[V11]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 3 : ui64, dSummary = "mgp.tensor_spec.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque) -> (!mgp.chain, !mgp.tensor_spec<f32>)
// CHECK-NEXT: [[V12:[^ ]*]] = mgp.tensor.create [[V3]], [[VARG1]][0], [[V9]], [[VOUTRESULTS_3]] : !mgp.tensor<f32>
// CHECK-NEXT: [[V13:[^ ]*]]:5 = grt.while ([[VARG0]]: !mef.chain, [[V8]]: !mgp.tensor<2xf32>, [[VARG1]]: !mgp.context, [[V4]]: !mgp.tensor<si64>, [[V12]]: !mgp.tensor<f32>) {
// CHECK-NEXT: [[V14:[^ ]*]] = mgp.tensor.extract.buffer [[VARG1]][0], [[V8]] : !mgp.tensor<2xf32>
// CHECK-NEXT: [[V15:[^ ]*]] = mgp.tensor.extract.buffer [[VARG1]][0], [[V4]] : !mgp.tensor<si64>
// CHECK-NEXT: [[V16:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V17:[^ ]*]] = mgp.device.context.extract [[V16]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_4:[^ ]*]], [[VOUTRESULTS_5:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V16]], [[V17]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 4 : ui64, dSummary = "mgp.tensor_spec.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque) -> (!mgp.chain, !mgp.tensor_spec<1xf32>)
// CHECK-NEXT: [[V18:[^ ]*]] = mgp.chain.host_to_device [[VARG0]], [[VARG1]][0]
// CHECK-NEXT: [[V19:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V20:[^ ]*]] = mgp.device.context.extract [[V19]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_6:[^ ]*]], [[VOUTRESULTS_7:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V19]], [[V20]], [[V18]], [[VARG1]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 5 : ui64, dSummary = "mgp.buffer.alloc.static", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque, !mgp.chain, !mgp.context) -> (!mgp.chain, !mgp.buffer<4>)
// CHECK-NEXT: [[V21:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V22:[^ ]*]] = mgp.device.context.extract [[V21]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_8:[^ ]*]], [[VOUTRESULTS_9:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V21]], [[V22]], [[VARG0]], [[VOUTRESULTS_7]], [[V14]], [[V15]])
// CHECK-NEXT: [[V23:[^ ]*]] = mgp.chain.host_to_device [[VOUTRESULTS_9]], [[VARG1]][0]
// CHECK-NEXT: [[V24:[^ ]*]] = mgp.tensor.create [[V23]], [[VARG1]][0], [[VOUTRESULTS_7]], [[VOUTRESULTS_5]] : !mgp.tensor<1xf32>
// CHECK-NEXT: [[V25:[^ ]*]] = mgp.tensor.extract.buffer [[VARG1]][0], [[V24]] : !mgp.tensor<1xf32>
// CHECK-NEXT: [[V26:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V27:[^ ]*]] = mgp.device.context.extract [[V26]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_10:[^ ]*]], [[VOUTRESULTS_11:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V26]], [[V27]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 3 : ui64, dSummary = "mgp.tensor_spec.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque) -> (!mgp.chain, !mgp.tensor_spec<f32>)
// CHECK-NEXT: [[V28:[^ ]*]] = mgp.tensor.create [[V18]], [[VARG1]][0], [[V25]], [[VOUTRESULTS_11]] : !mgp.tensor<f32>
// CHECK-NEXT: [[V29:[^ ]*]] = mgp.tensor.extract.buffer [[VARG1]][0], [[V12]] : !mgp.tensor<f32>
// CHECK-NEXT: [[V30:[^ ]*]] = mgp.tensor.extract.buffer [[VARG1]][0], [[V28]] : !mgp.tensor<f32>
// CHECK-NEXT: [[V31:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V32:[^ ]*]] = mgp.device.context.extract [[V31]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_12:[^ ]*]], [[VOUTRESULTS_13:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V31]], [[V32]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 7 : ui64, dSummary = "mgp.tensor_spec.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque) -> (!mgp.chain, !mgp.tensor_spec<bool>)
// CHECK-NEXT: [[V33:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V34:[^ ]*]] = mgp.device.context.extract [[V33]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_14:[^ ]*]], [[VOUTRESULTS_15:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V33]], [[V34]], [[V18]], [[VARG1]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 8 : ui64, dSummary = "mgp.buffer.alloc.static", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque, !mgp.chain, !mgp.context) -> (!mgp.chain, !mgp.buffer<1>)
// CHECK-NEXT: [[V35:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V36:[^ ]*]] = mgp.device.context.extract [[V35]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_16:[^ ]*]], [[VOUTRESULTS_17:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V35]], [[V36]], [[VARG0]], [[VOUTRESULTS_15]], [[V29]], [[V30]])
// CHECK-NEXT: [[V37:[^ ]*]] = mgp.chain.host_to_device [[VOUTRESULTS_17]], [[VARG1]][0]
// CHECK-NEXT: [[V38:[^ ]*]] = mgp.tensor.create [[V37]], [[VARG1]][0], [[VOUTRESULTS_15]], [[VOUTRESULTS_13]] : !mgp.tensor<bool>
// CHECK-NEXT: [[V39:[^ ]*]] = mgp.tensor.extract.buffer [[VARG1]][0], [[V38]] : !mgp.tensor<bool>
// CHECK-NEXT: [[V40:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V41:[^ ]*]] = mgp.device.context.extract [[V40]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_18:[^ ]*]], [[VOUTRESULTS_19:[^ ]*]] = "mgp.generic.execute"([[VARG1]], [[V40]], [[V41]], [[V18]], [[VARG1]], [[V39]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 10 : ui64, dSummary = "mgp.buffer.to_bool", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque, !mgp.chain, !mgp.context, !mgp.buffer<1>) -> (!mgp.chain, i1)
// CHECK-NEXT: mef.output [[VARG0]], [[V8]], [[VARG1]], [[V4]], [[V12]], [[VOUTRESULTS_19]] : !mef.chain, !mgp.tensor<2xf32>, !mgp.context, !mgp.tensor<si64>, !mgp.tensor<f32>, i1
// CHECK-NEXT: } do {
// CHECK-NEXT: [[V14:[^ ]*]] = mgp.tensor.extract.buffer [[VARG1]][0], [[V8]] : !mgp.tensor<2xf32>
// CHECK-NEXT: [[V15:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V16:[^ ]*]] = mgp.device.context.extract [[V15]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_4]], [[VOUTRESULTS_5]] = "mgp.generic.execute"([[VARG1]], [[V15]], [[V16]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 2 : ui64, dSummary = "mgp.tensor_spec.create", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque) -> (!mgp.chain, !mgp.tensor_spec<2xf32>)
// CHECK-NEXT: [[V17:[^ ]*]] = mgp.chain.host_to_device [[VARG0]], [[VARG1]][0]
// CHECK-NEXT: [[V18:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V19:[^ ]*]] = mgp.device.context.extract [[V18]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_6]], [[VOUTRESULTS_7]] = "mgp.generic.execute"([[VARG1]], [[V18]], [[V19]], [[V17]], [[VARG1]]) {aRuntimeSlot = 0 : ui64, bModuleIndex = 0 : ui64, cFunctionIndex = 11 : ui64, dSummary = "mgp.buffer.alloc.static", eIsAsync = false, fHasCuda = false, gCudaSlot = 0 : ui64} : (!mgp.context, !mef.chain, !mgp.opaque, !mgp.chain, !mgp.context) -> (!mgp.chain, !mgp.buffer<8>)
// CHECK-NEXT: [[V20:[^ ]*]] = grt.chain.create ()
// CHECK-NEXT: [[V21:[^ ]*]] = mgp.device.context.extract [[V20]], [[VARG1]][0] : !mgp.opaque
// CHECK-NEXT: [[VOUTCHAIN_8]], [[VOUTRESULTS_9]] = "mgp.generic.execute"([[VARG1]], [[V20]], [[V21]], [[VARG0]], [[VOUTRESULTS_7]], [[V14]], [[V14]])
// CHECK-NEXT: [[V22:[^ ]*]] = mgp.chain.host_to_device [[VOUTRESULTS_9]], [[VARG1]][0]
// CHECK-NEXT: [[V23:[^ ]*]] = mgp.tensor.create [[V22]], [[VARG1]][0], [[VOUTRESULTS_7]], [[VOUTRESULTS_5]] : !mgp.tensor<2xf32>
// CHECK-NEXT: mef.output [[VARG0]], [[V23]], [[VARG1]], [[V4]], [[V12]] : !mef.chain, !mgp.tensor<2xf32>, !mgp.context, !mgp.tensor<si64>, !mgp.tensor<f32>
// CHECK-NEXT: }
// CHECK-NEXT: mef.output [[V13]]#0, [[V13]]#1 : !mef.chain, !mgp.tensor<2xf32>
// CHECK-NEXT: }

mo.graph @while_op() -> (!mo.tensor<[2], f32>) {
   %x = mo.constant { value = #M.dense_array<1.0, 1.0> : tensor<2xf32> } : !mo.tensor<[2], f32>
   %y = mo.constant { value = #M.dense_array<4.0> : tensor<1xf32> } : !mo.tensor<[], f32>
   %res = mo.while (%x as %xi: !mo.tensor<[2], f32>) {
     %zero = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
     %mean0 = mo.mean (%xi, %zero) : (!mo.tensor<[2], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1], f32>
     %shape = mo.constant {value = #M.dense_array<> : tensor<0xsi64>} : !mo.tensor<[0], si64>
     %mean1 = mo.reshape (%mean0, %shape) : (!mo.tensor<[1], f32>, !mo.tensor<[0], si64>) -> !mo.tensor<[], f32>
     %cond = mo.greater (%y, %mean1) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>) -> !mo.tensor<[], bool>
     mo.yield %cond : !mo.tensor<[], bool>
   } do {
     %new_x = mo.add (%xi, %xi) : !mo.tensor<[2], f32>
     mo.yield %new_x : !mo.tensor<[2], f32>
   }
   mo.output %res : !mo.tensor<[2], f32>
}
