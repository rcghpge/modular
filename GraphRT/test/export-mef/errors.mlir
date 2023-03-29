// RUN: mef-translate -export-mef -allow-unregistered-dialect -split-input-file --verify-diagnostics  %s

// expected-error @+1 {{expected non-empty function body}}
mef.invokable @trivial() {
}

// -----

// expected-error @+2 {{expects regions to end with 'mef.output', found 'unknown.term'}}
// expected-note @+1 {{the absence of terminator implies 'mef.output'}}
mef.invokable @need_output() {
  "unknown.term"() : () -> ()
}

// -----

mef.invokable @need_output() -> i32 {
  // expected-error @+1 {{'mef.output' op expected 1 operands for enclosing op}}
  mef.output
}

// -----

mef.invokable @type_mismatch(%a: f32) -> i32 {
  // expected-error @+1 {{'mef.output' op operand #0 has type 'f32' but should be 'i32'}}
  mef.output %a : f32
}

// -----

mef.invokable @type_attr_signless() {
  // expected-error @+1 {{signless integers not supported: 'i32'}}
  "foo"() {t = i32} : () -> ()
  mef.output
}

// -----

mef.invokable @type_attr_non_whole_bytes_int() {
  // expected-error @+1 {{integer cannot be converted to MEF type: 'si13'}}
  "foo"() {t = si13} : () -> ()
  mef.output
}

// -----

mef.invokable @type_quantized_per_axis() {
  // FIXME: We currently have a hack for this.
  // xpected-error @+1 {{per-axis quantized type not supported: '!quant.uniform<i8:f32:0, {1.000000e-01:-128}>'}}
  "foo"() {t = !quant.uniform<i8:f32:0, {0.1:-128}>} : () -> ()
  mef.output
}
