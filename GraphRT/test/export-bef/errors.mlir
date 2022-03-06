// RUN: bef-translate -export-bef -allow-unregistered-dialect -split-input-file --verify-diagnostics  %s 

bef.func @trivial() {  // expected-error {{must have exactly one block}}
}

// -----

// expected-error @+2 {{expects regions to end with 'bef.output', found 'unknown.term'}}
// expected-note @+1 {{the absence of terminator implies 'bef.output'}}
bef.func @need_output() {
  "unknown.term"() : () -> ()
}

// -----

bef.func @need_output() -> i32 {
  bef.output  // expected-error {{'bef.output' op expected 1 operands for enclosing op}}
}

// -----

bef.func @type_mismatch(%a: f32) -> i32 {
  // expected-error @+1 {{'bef.output' op operand #0 has type 'f32' but should be 'i32'}}
  bef.output %a : f32
}
