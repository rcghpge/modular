// RUN: mef-translate -export-mef -allow-unregistered-dialect %s -o %t
// RUN: mef-translate -import-mef %t | FileCheck %s

// CHECK-LABEL: "mef.invokable" ["dict"]
mef.invokable @dict() {
  %ch5 = grt.chain.create ()
  // CHECK: "test.print_all_dict" [{aString = "stringAttr", theNumber3 = 3 : si32, theNumber5 = 5 : ui32}]
  %ch6 = "test.print_all_dict"(%ch5) { value = { theNumber3 = 3 : si32, theNumber5 = 5 : ui32, aString = "stringAttr" } } : (!mef.chain) -> (!mef.chain)

  mef.output
}

// CHECK-LABEL: "mef.invokable" ["nested_dict"]
mef.invokable @nested_dict() {
  %ch5 = grt.chain.create ()
  // CHECK: "test.print_all_dict" [{aDict = {aString = "stringAttr"}, theNumber3 = 3 : si32, theNumber5 = 5 : ui32}]
  %ch6 = "test.print_all_dict"(%ch5) { value = { theNumber3 = 3 : si32, theNumber5 = 5 : ui32, aDict = { aString = "stringAttr" } } } : (!mef.chain) -> (!mef.chain)
  mef.output
}
