// RUN: graphrt-opt -allow-unregistered-dialect -split-input-file --verify-diagnostics %s

// CHECK-LABEL: --- Running 'trivial':
mef.invokable @trivial() {
  // expected-error @+1 {{'grt.invoke' op '@foo' does not reference a valid invokable}}
  grt.invoke @foo() : () -> ()
  mef.output
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid1':
mef.invokable @controlflow_while_invalid1() {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 4
  %res:2 = grt.while (%i: si32, %n: si32) {
    %bogus = grt.constant.si32 42
    // expected-error @+1 {{'mef.output' op operand #0 has type 'si32' but expected 'i1'}}
    mef.output %bogus: si32
  } do {
    mef.output %i, %n: si32, si32
  }
  mef.output %res#0, %res#1: si32, si32
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid2':
mef.invokable @controlflow_while_invalid2() {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 4
  %res:2 = grt.while (%i: si32, %n: si32) {
    %cond = grt.lessequal.si32 %i, %n
    mef.output %cond: i1
  } do {
    // expected-error @+1 {{'mef.output' op expected 2 operands, but given 1}}
    mef.output %i: si32
  }
  mef.output %res#0, %res#1: si32, si32
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid3':
mef.invokable @controlflow_while_invalid3() {
  %i = grt.constant.si32 0
  %n = grt.constant.si32 4
  %res:2 = grt.while (%i: si32, %n: si32) {
    %cond = grt.lessequal.si32 %i, %n
    mef.output %cond: i1
  } do {
    %bogus = grt.lessequal.si32 %i, %n
    // expected-error @+1 {{'mef.output' op operand #1 has type 'i1' but expected 'si32'}}
    mef.output %i, %bogus: si32, i1
  }
  mef.output %res#0, %res#1: si32, si32
}

// -----

// CHECK-LABEL: --- Running 'controlflow_while_invalid4':
mef.invokable @controlflow_while_invalid4() {
  %i = grt.constant.si32 0
  %bogus = grt.constant.si32 42
  %res = grt.while (%i: si32) {
    %n = grt.constant.si32 4
    %cond = grt.lessequal.si32 %i, %n
    mef.output %cond: i1
  } do {
    // expected-error @+1 {{use of undeclared SSA value name}}
    %new_i = grt.add.si32 %i, %bogus
    mef.output %new_i: si32
  }
  mef.output %res: si32
}
