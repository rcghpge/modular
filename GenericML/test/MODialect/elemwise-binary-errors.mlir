// RUN: mo-opt %s --split-input-file --verify-diagnostics

// expected-note@+1 {{prior use here}}
mo.graph @add(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x5xf32>) -> (!mo.tensor<2x3xf32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @sub(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x10xf32>) -> (!mo.tensor<2x3xf32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.sub(%arg0, %arg1) : !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @mul(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x10xf32>) -> (!mo.tensor<2x3xf32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.mul(%arg0, %arg1) : !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// -----

// expected-note@+1 {{prior use here}}
mo.graph @div(%arg0: !mo.tensor<2x3xf32>, %arg1: !mo.tensor<2x10xf32>) -> (!mo.tensor<2x3xf32>) {
  // expected-error@+1 {{use of value '%arg1' expects different type than prior uses}}
  %0 = mo.div(%arg0, %arg1) : !mo.tensor<2x3xf32>
  mo.output %0 : !mo.tensor<2x3xf32>
}

// -----

// expected-error@+1 {{failed to parse MO_Tensor parameter 'dtype' which is to be a `DType}}
mo.graph @signless(%arg0: !mo.tensor<i32>) -> (!mo.tensor<i32>) {
  mo.output %arg0 : !mo.tensor<i32>
}

// -----

// TODO relocate softmax test cases to a proper category
mo.graph @softmax(%arg0: !mo.tensor<3xf32>) -> (!mo.tensor<3xf32>) {
  // expected-error@+1 {{'mo.softmax' op operand #0 must be dim 2 or unranked, but got '!mo.tensor<3xf32>'}}
  %0 = mo.softmax(%arg0) : (!mo.tensor<3xf32>) -> !mo.tensor<3xf32>
  mo.output %0 : !mo.tensor<3xf32>
}
