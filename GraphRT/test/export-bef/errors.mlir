// RUN: bef-translate -export-bef -allow-unregistered-dialect -split-input-file --verify-diagnostics  %s

// expected-error @+1 {{expected non-empty function body}}
bef.invokable @trivial() {
}

// -----

// expected-error @+2 {{expects regions to end with 'bef.output', found 'unknown.term'}}
// expected-note @+1 {{the absence of terminator implies 'bef.output'}}
bef.invokable @need_output() {
  "unknown.term"() : () -> ()
}

// -----

bef.invokable @need_output() -> i32 {
  // expected-error @+1 {{'bef.output' op expected 1 operands for enclosing op}}
  bef.output
}

// -----

bef.invokable @type_mismatch(%a: f32) -> i32 {
  // expected-error @+1 {{'bef.output' op operand #0 has type 'f32' but should be 'i32'}}
  bef.output %a : f32
}

// -----

bef.invokable @type_attr_signless() {
  // expected-error @+1 {{signless integers not supported: 'i32'}}
  "foo"() {t = i32} : () -> ()
  bef.output
}

// -----

bef.invokable @type_attr_non_whole_bytes_int() {
  // expected-error @+1 {{integer cannot be converted to BEF type: 'si13'}}
  "foo"() {t = si13} : () -> ()
  bef.output
}

// -----

bef.invokable @type_quantized_per_axis() {
  // FIXME: We currently have a hack for this.
  // xpected-error @+1 {{per-axis quantized type not supported: '!quant.uniform<i8:f32:0, {1.000000e-01:-128}>'}}
  "foo"() {t = !quant.uniform<i8:f32:0, {0.1:-128}>} : () -> ()
  bef.output
}
