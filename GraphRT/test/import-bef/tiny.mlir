// RUN: bef-translate -export-bef -allow-unregistered-dialect %s -o %t
// RUN: bef-translate -import-bef -allow-unregistered-dialect %t | FileCheck %s

// CHECK: module {

// CHECK-LABEL: "bef.invokable" ["trivial"]
// CHECK-NEXT:    %{{.*}}:2 = bef.opaque "simple.primitive" []() : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    bef.opaque "grt.invoke" [@trivial]() : () -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
bef.invokable @trivial() {
  %a, %b = "simple.primitive"() : () -> (i8, i16)
  grt.invoke @trivial() : () -> ()
  bef.output
}

// CHECK-NEXT: }
