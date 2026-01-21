// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --split-input-file --verify-diagnostics

kgen.func @foo() -> si64 {
  %c = mosh.param.to_value = <42>
  kgen.return %c : si64
}

mo.graph @unsupported_expr() {
  // expected-error@+1 {{cannot lower parameter expression}}
  kgen.param.declare fn: () -> si64 = <@foo>
  mo.output
}

// -----

mo.graph @unsupported_poc<X>(%arg0: !mo.tensor<[X], f32>) {
  // expected-error@+1 {{unsupported POC 'in'}}
  kgen.param.declare x: i1 = <in(:si64 X, [2, 4])>
  mo.output
}
