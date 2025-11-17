// RUN: mo-opt -verify-diagnostics %s

!tt = !mo.tensor<[10, 10], f32>

mogg.kernel.stub @stub_0(%arg0: !mo.tensor<[], f32>) {
  mogg.output
}

mo.graph @too_few_args() -> () {
  mogg.experimental.kernel () -> () {
    // expected-error@below{{Callee expected 1 arguments but got 0}}
    mogg.call.stub @stub_0
    mogg.call.execute["foo"]
  } {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

mo.graph @too_many_args() -> () {
  mogg.experimental.kernel () -> () {
    %0:2 = mogg.output_placeholders : !mo.tensor<[], f32>, !mo.tensor<[], f32>
    // expected-error@below{{Callee expected 1 arguments but got 2}}
    mogg.call.stub @stub_0 %0#0, %0#1 : !mo.tensor<[], f32>, !mo.tensor<[], f32>
    mogg.call.execute["foo"]
  } {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

mo.graph @mismatching_args() -> () {
  mogg.experimental.kernel () -> () {
    %0 = mogg.output_placeholders : !mo.tensor<[], f64>
    // expected-error@below{{Callee argument at position 0 expected to be !mo.tensor<[], f32> but got !mo.tensor<[], f64>}}
    mogg.call.stub @stub_0 %0 : !mo.tensor<[], f64>
    mogg.call.execute["foo"]
  } {device = #M.device_ref<"cpu", 0>}
  mo.output
}

// -----

mo.graph @does_not_exist() -> () {
  mogg.experimental.kernel () -> () {
    // expected-error@below{{There is no kernel stub named 'stub_that_does_not_exist'}}
    mogg.call.stub @stub_that_does_not_exist
    mogg.call.execute["foo"]
  } {device = #M.device_ref<"cpu", 0>}
  mo.output
}
