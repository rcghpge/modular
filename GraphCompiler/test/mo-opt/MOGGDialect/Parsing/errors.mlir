// RUN: mo-opt -split-input-file -verify-diagnostics %s

!tt = !mo.tensor<[10, 10], f32>

mogg.kernel.stub @stub_0(%arg0: !mo.tensor<[], f32>) fusionInfo = "stub_0"
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}

mo.graph @too_few_args() -> () {
  mogg.kernel () -> () {
    // expected-error@below{{Kernel stub 'stub_0' expects 1 arguments, but got 0}}
    mogg.call.stub @stub_0
    mogg.call.execute["foo"] {moOpName = "mo.custom"}
  } fusionInfo = "foo" {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

mogg.kernel.stub @stub_0(%arg0: !mo.tensor<[], f32>) fusionInfo = "stub_0"
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}

mo.graph @too_many_args() -> () {
  mogg.kernel () -> () {
    %0:2 = mogg.output_placeholders : !mo.tensor<[], f32>, !mo.tensor<[], f32>
    // expected-error@below{{Kernel stub 'stub_0' expects 1 arguments, but got 2}}
    mogg.call.stub @stub_0 %0#0, %0#1 : !mo.tensor<[], f32>, !mo.tensor<[], f32>
    mogg.call.execute["foo"] {moOpName = "mo.custom"}
  } fusionInfo = "foo" {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

mogg.kernel.stub @stub_0(%arg0: !mo.tensor<[], f32>) fusionInfo = "stub_0"
  attributes {abi = 0 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}

mo.graph @mismatching_args() -> () {
  mogg.kernel () -> () {
    %0 = mogg.output_placeholders : !mo.tensor<[], f64>
    // expected-error@below{{Kernel stub 'stub_0' expects argument 0 to have type !mo.tensor<[], f32>, but got !mo.tensor<[], f64>}}
    mogg.call.stub @stub_0 %0 : !mo.tensor<[], f64>
    mogg.call.execute["foo"] {moOpName = "mo.custom"}
  } fusionInfo = "foo" {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

mo.graph @does_not_exist() -> () {
  mogg.kernel () -> () {
    // expected-error@below{{There is no kernel stub named 'stub_that_does_not_exist'}}
    mogg.call.stub @stub_that_does_not_exist
    mogg.call.execute["foo"] {moOpName = "mo.custom"}
  } fusionInfo = "foo" {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

mogg.kernel.stub @stub_1() fusionInfo = "stub_1"
  attributes {abi = 1 : i32, constantResources = [], device = #M.device_ref<"cpu", 0>, isAsync = true} {
  mogg.output
}

mo.graph @wrong_abi(%0 : !mo.tensor<[], f32>) -> () {
  mogg.kernel () -> () {
    // expected-error@below{{Call site should be in the MOGG ABI}}
    mogg.call.stub @stub_1
    mogg.call.execute["foo"] {moOpName = "mo.custom"}
  } fusionInfo = "foo" {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

// mogg.call.build_graph must reference a kernel.decl op.
func.func @build_graph_missing_decl(
    %in: !mo.tensor<[2, 4], f32, gpu:0>,
    %b: !mgp.device_graph.builder) {
  // expected-error@+1 {{'missing.kernel' does not refer to a kernel.decl op}}
  mogg.call.build_graph["missing.kernel"]
      inputs(%in : !mo.tensor<[2, 4], f32, gpu:0>)
      builder(%b) {moOpName = "mo.matmul"}
  return
}
