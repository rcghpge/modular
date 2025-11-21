// RUN: EMIT_MOJO=1 mo-opt -merge-identical-kernel-stubs

mogg.kernel.stub @func_1()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_1"} {
  mogg.output
}

mogg.kernel.stub @func_2()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_2"} {
  mogg.output
}

// CHECK-LABEL: @func_3
// CHECK:      mogg.call.stub @func_1
// CHECK-NEXT: mogg.call.stub @func_1
// CHECK-NOT:  mogg.call.stub @func
mogg.kernel.stub @func_3()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_3"} {
  mogg.call.stub @func_1
  mogg.call.stub @func_2
  mogg.output
}

// CHECK-LABEL: @func_4
// CHECK:      mogg.call.stub @func_1
// CHECK:      mogg.call.stub @func_1
// CHECK-NOT:  mogg.call.stub @func
mogg.kernel.stub @func_4()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_4"} {
  mogg.call.stub @func_1
  mogg.call.stub @func_2
  mogg.output
}

// CHECK-LABEL: @func_5
// CHECK:      mogg.call.stub @func_3
// CHECK:      mogg.call.stub @func_3
// CHECK-NOT:  mogg.call.stub @func
mogg.kernel.stub @func_5()
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true, summary = "func_5"} {
  mogg.call.stub @func_3
  mogg.call.stub @func_4
  mogg.output
}
