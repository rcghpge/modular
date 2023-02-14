// RUN: graphrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

mef.invokable @trivial() {
  // expected-error @+1 {{'@foo' does not reference a valid invokable}}
  grt.invoke @foo() : () -> ()
  mef.output
}
