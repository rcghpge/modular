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

mo.graph @unsupported_poc<X>(%arg0: !mo.tensor<[X], f32>) {
  // expected-error@+1 {{unsupported POC 'in'}}
  kgen.param.declare x: i1 = <in(X, [2, 4])>
  mo.output
}
