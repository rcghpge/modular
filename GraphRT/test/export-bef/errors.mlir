// RUN: bef-translate -export-bef -allow-unregistered-dialect -split-input-file --verify-diagnostics  %s

bef.invokable @trivial() {  // expected-error {{expected non-empty function body}}
}

// -----

// expected-error @+2 {{expects regions to end with 'bef.output', found 'unknown.term'}}
// expected-note @+1 {{the absence of terminator implies 'bef.output'}}
bef.invokable @need_output() {
  "unknown.term"() : () -> ()
}

// -----

bef.invokable @need_output() -> i32 {
  bef.output  // expected-error {{'bef.output' op expected 1 operands for enclosing op}}
}

// -----

bef.invokable @type_mismatch(%a: f32) -> i32 {
  // expected-error @+1 {{'bef.output' op operand #0 has type 'f32' but should be 'i32'}}
  bef.output %a : f32
}

// -----

bef.invokable @type_attr_signless() {
  // expected-error @+1 {{signless and arbitrary length integers not supported yet: 'i32'}}
  "foo"() {t = i32} : () -> ()
  bef.output
}

// -----

bef.invokable @type_attr_non_whole_bytes_int() {
  // expected-error @+1 {{signless and arbitrary length integers not supported yet: 'si13'}}
  "foo"() {t = si13} : () -> ()
  bef.output
}

// -----

bef.invokable @type_attr_too_long_int() {
  // expected-error @+1 {{signless and arbitrary length integers not supported yet: 'si256'}}
  "foo"() {t = si256} : () -> ()
  bef.output
}

// -----

bef.invokable @type_attr_tensor() {
  // expected-error @+1 {{type attribute not supported: 'tensor<1x3xf32>'}}
  "foo"() {t = tensor<1x3xf32>} : () -> ()
  bef.output
}
