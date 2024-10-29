// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t.mef

// This test is a little ugly because of we need to check for more than 256
// arguments to a function type. This function has 257 arguments. On failure
// we get a "wrapping" behavior, where MEF thinks it has only one argument. So
// if the output looks like this:
//
// Type attribute: (si32) -> si32
//
// that indicates the failure case.

grt.testcase @test_function_type_with_257_args{
  %ch0 = grt.chain.create ()
  // We need to have one _long_ line here for the CHECK - sorry!
  // CHECK: Type attribute: (si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32, si32) -> si32
  %ch1 = "test.print_type"(%ch0) {t = (
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32, si32, si32, si32, si32, si32, si32, si32,
    si32) -> (si32) } : (!mef.chain) -> (!mef.chain)
  mef.output
}

