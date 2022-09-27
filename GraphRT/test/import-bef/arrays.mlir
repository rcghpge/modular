// RUN: bef-translate -export-bef -allow-unregistered-dialect %s -o %t
// RUN: bef-translate -import-bef -allow-unregistered-dialect %t | FileCheck %s

bef.invokable @attributes() {
  // CHECK: "a.b" [array<ui32: 1, 2, 3, 4>]
  "a.b"() { a = dense<[[1, 2], [3, 4]]> : tensor<2x2xui32> } : () -> ()

  // CHECK-NEXT: "a.b" [array<ui64: 42>]
  "a.b"() { a = array<ui64: 42> }  : () -> ()

  // CHECK-NEXT: "a.b" [array<i32>]
  "a.b"() { a = array<i32> } : () -> ()

  // CHECK-NEXT: "a.b" [array<i64: 0, 1, 2>]
  "a.b"() { a = array<i64: 0, 1, 2> } : () -> ()

  // CHECK-NEXT: "a.b" [array<f32: 1.500000e+00, 2.500000e+00, 3.500000e+00>]
  "a.b"() { a = array<f32: 1.5, 2.5, 3.5> } : () -> ()

  // CHECK-NEXT: "a.b" [array<i1: true, false, true>]
  "a.b"() { a = array<i1: true, false, true> }: () -> ()

  // CHECK-NEXT: "a.b" [dense_resource<befArray> : tensor<1024xui32>]
  "a.b"() { a = dense<0> : tensor<1024xui32> } : () -> ()

  // Small arrays get inlined.
  // CHECK-NEXT: "a.b" [array<ui64: 1, 2, 3>]
  "a.b"() { a = dense_resource<smallResource> : tensor<3xui64> } : () -> ()

  bef.output
}

// Check that the big arrays were properly transformed into a resource.

// CHECK: dialect_resources: {
// CHECK-NEXT:   builtin: {
// CHECK-NEXT:     befArray: "0x01{{0+}}"

// Input resources for the tests above.
{-#
  dialect_resources: {
    builtin: {
      smallResource: "0x08000000010000000000000002000000000000000300000000000000"
    }
  }
#-}
