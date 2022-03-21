// RUN: graphrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

bef.invokable @trivial() {  
  grt.invoke @foo() : () -> () // expected-error {{'@foo' does not reference a valid invokable}}
  bef.output
}
