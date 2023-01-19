// RUN: mo-opt %s --split-input-file --verify-diagnostics

// expected-note@+1 {{prior use here}}
mo.graph @abs(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.abs(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @relu(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.relu(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @gelu(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.gelu(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @erf(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.erf(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @exp(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.exp(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @tanh(%arg0: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[4], f32>) {
  // expected-error@+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = mo.tanh(%arg0) : !mo.tensor<[4], f32>
  mo.output %0 : !mo.tensor<[4], f32>
}
