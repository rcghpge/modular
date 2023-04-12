// RUN: mo-opt %s --mo-to-mogg --mo-to-primitives --split-input-file --verify-diagnostics --allow-unregistered-dialect

mo.graph @duplicate_param_decl<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  kgen.param.declare Z = <X>
  // expected-error@+1 {{Duplicate declaration of Z}}
  kgen.param.declare Z = <Y>
  mo.output
}

// -----

mo.graph @unsupported_expr() {
  // expected-error@+1 {{cannot lower parameter expression}}
  kgen.param.declare fn: () -> index = <@foo>
  mo.output
}

// -----

mo.graph @unsupported_poc<X, Y>(%arg0: !mo.tensor<[X, Y], f32>) {
  // expected-error@+1 {{unsupported POC 'mod'}}
  kgen.param.declare Z = <mod(X, Y)>
  mo.output
}
