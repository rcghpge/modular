// RUN: mo-opt %s --split-input-file --verify-diagnostics


mo.graph @gelu(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) attributes {isSubgraph} {
  %0 = mo.gelu(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

mo.graph @main(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // expected-error@+1 {{Cannot find graph for symbol @gelu_notfound}}
  %0 = mo.call @gelu_notfound (%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// -----

mo.graph @gelu(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  %0 = mo.gelu(%arg0) : !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

mo.graph @main(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[4, 5], f32>) {
  // expected-error@+1 {{Only subgraphs can be called}}
  %0 = mo.call @gelu (%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[4, 5], f32>
  mo.output %0 : !mo.tensor<[4, 5], f32>
}

// -----

// expected-note@+1 {{graph op here}}
mo.graph @gelu(%arg0: !mo.tensor<[6, 5], f32>) -> (!mo.tensor<[6, 5], f32>) attributes {isSubgraph} {
  %0 = mo.gelu(%arg0) : !mo.tensor<[6, 5], f32>
  mo.output %0 : !mo.tensor<[6, 5], f32>
}

mo.graph @main(%arg0: !mo.tensor<[4, 5], f32>) -> (!mo.tensor<[6, 5], f32>) {
  // expected-error@+1 {{Subgraph gelu has wrong type for argument 0}}
  %0 = mo.call @gelu (%arg0) : (!mo.tensor<[4, 5], f32>) -> !mo.tensor<[6, 5], f32>
  mo.output %0 : !mo.tensor<[6, 5], f32>
}
