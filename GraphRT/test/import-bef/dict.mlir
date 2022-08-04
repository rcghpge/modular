// RUN: bef-translate -export-bef -allow-unregistered-dialect %s -o %t
// RUN: bef-translate -import-bef %t | FileCheck %s

// CHECK-LABEL: "bef.invokable" ["dict"]
bef.invokable @dict() {
  %ch5 = grt.chain.new
  // CHECK: "test.print_all_dict" [{aString = "stringAttr", theNumber3 = 3 : si32, theNumber5 = 5 : ui32}]
  %ch6 = "test.print_all_dict"(%ch5) { value = { theNumber3 = 3 : si32, theNumber5 = 5 : ui32, aString = "stringAttr" } } : (!bef.chain) -> (!bef.chain)

  bef.output
}

// CHECK-LABEL: "bef.invokable" ["nested_dict"]
bef.invokable @nested_dict() {
  %ch5 = grt.chain.new
  // CHECK: "test.print_all_dict" [{aDict = {aString = "stringAttr"}, theNumber3 = 3 : si32, theNumber5 = 5 : ui32}]
  %ch6 = "test.print_all_dict"(%ch5) { value = { theNumber3 = 3 : si32, theNumber5 = 5 : ui32, aDict = { aString = "stringAttr" } } } : (!bef.chain) -> (!bef.chain)
  bef.output
}
