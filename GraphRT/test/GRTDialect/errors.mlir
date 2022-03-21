// RUN: graphrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

bef.invokable @trivial() {  
  grt.call @foo() : () -> () // expected-error {{'@foo' does not reference a valid function}}
  bef.output
}
