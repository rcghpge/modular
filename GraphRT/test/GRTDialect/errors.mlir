// RUN: graphrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

bef.invokable @trivial() {
  // expected-error @+1 {{'@foo' does not reference a valid invokable}}
  grt.invoke @foo() : () -> ()
  bef.output
}
