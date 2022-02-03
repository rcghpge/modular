// RUN: graphrt-translate -export-bef -split-input-file --verify-diagnostics  %s 


func @trivial() { // expected-error {{empty block: expect at least a terminator}}
^bb1:
  //bef.return

^bb2:
  //bef.return
}

// -----

// expected-error @+1 {{external functions are not allowed}}
func private @external_func() -> i32
