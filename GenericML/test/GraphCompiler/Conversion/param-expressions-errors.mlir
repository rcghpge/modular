// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --split-input-file --verify-diagnostics --allow-unregistered-dialect

kgen.func @foo() -> index {
  %c = index.constant 42
  kgen.return %c : index
}

mo.graph @unsupported_expr() {
  // expected-error@+1 {{cannot lower parameter expression}}
  kgen.param.declare fn: () -> index = <@foo>
  mo.output
}

// -----

mo.graph @unsupported_poc<X, Y, B:i1>(%arg0: !mo.tensor<[X, Y], f32>) {
  // expected-error@+1 {{unsupported POC 'cond'}}
  kgen.param.declare Z = <cond(B, 1, 2)>
  mo.output
}
