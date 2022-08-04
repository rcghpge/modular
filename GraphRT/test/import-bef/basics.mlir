// RUN: bef-translate -export-bef -allow-unregistered-dialect %s -o %t
// RUN: bef-translate -import-bef -allow-unregistered-dialect %t | FileCheck %s

// RUN: bef-translate -export-bef -allow-unregistered-dialect -enable-bef-metadata %s -o %t2
// RUN: bef-translate -import-bef -allow-unregistered-dialect %t2 | FileCheck %s --check-prefix=MD

// Check to make sure the magic number is right, and the first section is there.

// CHECK: module {

// CHECK-LABEL: "bef.invokable" ["trivial"]
// CHECK-NEXT:    %0:2 = bef.opaque "simple.primitive" []() : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    bef.opaque "grt.invoke" [@trivial]() : () -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
// MD-LABEL: "bef.invokable" ["trivial"]
// MD-NEXT:    %0:2 = "simple.primitive"() : () -> (i8, i16)
// MD-NEXT:    "grt.invoke"() {target = @trivial} : () -> ()
// MD-NEXT:    bef.output
// MD-NEXT:  }
bef.invokable @trivial() {
  %a, %b = "simple.primitive"() : () -> (i8, i16)
  grt.invoke @trivial() : () -> ()
  bef.output
}

// CHECK-LABEL:  "bef.invokable" ["args_results"]
// CHECK-NEXT:   ^bb0(%arg0: !bef.unknown, %arg1: !bef.unknown):
// CHECK-NEXT:     bef.output %arg1, %arg0 : !bef.unknown, !bef.unknown
// CHECK-NEXT:   }
// MD-LABEL:  "bef.invokable" ["args_results"]
// MD-NEXT:   ^bb0(%arg0: si32, %arg1: si8):
// MD-NEXT:     bef.output %arg1, %arg0 : si8, si32
// MD-NEXT:   }
bef.invokable @args_results(%a: si32, %b: si8) -> (si8, si32) {
  bef.output %b, %a : si8, si32
}

// CHECK-LABEL:  "bef.invokable" ["attributes"]
// MD-LABEL:  "bef.invokable" ["attributes"]
bef.invokable @attributes() {
  // CHECK-NEXT: "a.b" [true]
  // MD-NEXT: "a.b"() {a = true}
  "a.b"() { a = 1 : i1 }: () -> ()

  // CHECK-NEXT: "a.b" [42 : si8, 17 : ui8]
  // MD-NEXT: "a.b"() {b = 42 : si8, x = 17 : ui8}
  "a.b"() { b = 42 : si8, x = 17 : ui8 }: () -> ()

  // CHECK-NEXT: "a.b" [258 : si16]
  // MD-NEXT: "a.b"() {c = 258 : si16}
  "a.b"() { c = 0x0102 : si16 }: () -> ()

  // CHECK-NEXT: "a.b" [32751241 : si32]
  "a.b"() { d = 32751241 : si32 }: () -> ()

   // CHECK-NEXT: "a.b" [3275124123121 : si64]
  "a.b"() { a = 3275124123121 : si64 }: () -> ()

  // CHECK-NEXT: "a.b" ["foo\00bar"]
  "a.b"() { a = "foo\00bar" : si64 }: () -> ()

  // CHECK-NEXT: "a.b" [dense<> : tensor<0xsi32>]
  "a.b"() { a = dense<[]> : tensor<0xsi32> }: () -> ()

  // CHECK-NEXT: "a.b" [dense<[true, false, true]> : tensor<3xi1>]
  "a.b"() { a = dense<[true, false, true]> : tensor<3xi1> }: () -> ()

  // CHECK-NEXT: "a.b" [2.000000e+00 : f32]
  "a.b"() { a = 2.0 : f32 }: () -> ()

  // CHECK-NEXT: "a.b" [3.000000e+00]
  "a.b"() { a = 3.0 : f64 }: () -> ()

  // Dictionary attribute.
  // CHECK-NEXT: bef.opaque "a.b" [{b = 1 : si8, c = [true, false, false]}]
  "a.b"() { a = { b = 1 : si8, c = [true, false, false] } }: () -> ()

  // List Attribute.
  // CHECK-NEXT: bef.opaque "a.b"  [1 : si8, [1 : ui32, 2 : si8]]
  // MD: "a.b"() {a = 1 : si8, b = [1 : ui32, 2 : si8]} : () -> ()
  "a.b"() {a = 1: si8, b = [1: ui32, 2: si8] } : () -> ()

  // Type attribute.
  // CHECK-NEXT: bef.opaque "type.i32" [si32]()
  // MD: "type.i32"() {t = si32}
  "type.i32"() {t = si32} : () -> ()

  // CHECK-NEXT: bef.opaque "type.f32" [f32]()
  // MD:  "type.f32"() {t = f32}
  "type.f32"() {t = f32} : () -> ()

  // CHECK-NEXT: bef.opaque "type.f128" [f128]()
  // MD: "type.f128"() {t = f128}
  "type.f128"() {t = f128} : () -> ()

  // CHECK-NEXT: bef.opaque "type.tensor" [tensor<4x1xf32>]()
  // MD: "type.tensor"() {t = tensor<4x1xf32>}
  "type.tensor"() {t = tensor<4x1xf32>} : () -> ()

  // CHECK-NEXT: bef.opaque "type.tensor" [tensor<0x1x0xcomplex<si32>>]()
  // MD: "type.tensor"() {t = tensor<0x1x0xcomplex<si32>>
  "type.tensor"() {t = tensor<0x1x0xcomplex<si32>>} : () -> ()

  // CHECK-NEXT: bef.opaque "type.quanttensor" [tensor<1x2x!quant.uniform<i16:f32:1, {0.20000000298023224:128,0.10000000149011612:-128}>>]()
  // MD: "type.quanttensor"() {t = tensor<1x2x!quant.uniform<i16:f32:1, {0.20000000298023224:128,0.10000000149011612:-128}>>}
  "type.quanttensor"() {t = tensor<1x2x!quant.uniform<i16:f32:1, {0.2:128, 0.1:-128}>>} : () -> ()

  // CHECK-NEXT: bef.opaque "type.quant" [!quant.uniform<u8:f32, 0.0078119998797774315:128>]()
  // MD: "type.quant"() {t = !quant.uniform<u8:f32, 0.0078119998797774315:128>
  "type.quant"() {t = !quant.uniform<u8:f32, 0.0078119998797774315:128>} : () -> ()

  // CHECK-NEXT: bef.opaque "type.quant" [!quant.uniform<i8:f32, 0.0078119998797774315:-128>]()
  // MD: "type.quant"() {t = !quant.uniform<i8:f32, 0.0078119998797774315:-128>
  "type.quant"() {t = !quant.uniform<i8:f32, 0.0078119998797774315:-128>} : () -> ()

  // CHECK-NEXT: bef.opaque "type.fn" [(si32, !quant.uniform<i8:f32, 0.0078119998797774315:-128>) -> tensor<4x100xf32>]()
  // MD: "type.fn"() {t = (si32, !quant.uniform<i8:f32, 0.0078119998797774315:-128>) -> tensor<4x100xf32>
  "type.fn"() {t = (si32, !quant.uniform<i8:f32, 0.0078119998797774315:-128>) -> tensor<4x100xf32>} : () -> ()

  // CHECK-NEXT: bef.opaque "type.fn" [() -> (si8, i1)]()
  // MD: "type.fn"() {t = () -> (si8, i1)
  "type.fn"() {t = () -> (si8, i1)} : () -> ()

  // We're dropping the quantization information here.
  // CHECK-NEXT: bef.opaque "type.calibrated" [tensor<1x28x28xf32>]()
  // MD: "type.calibrated"() {t = tensor<1x28x28xf32>
  "type.calibrated"() {t = tensor<1x28x28x!quant.calibrated<f32<0.000000e+00:2.550000e+02>>>} : () -> ()

  // Check that we correctly serialize and deserialize the chain type.
  // CHECK-NEXT: bef.opaque "type.chain" [!bef.chain]()
  // MD: "type.chain"() {t = !bef.chain
  "type.chain"() {t = !bef.chain} : () -> ()

  bef.output
   // CHECK-NEXT:     bef.output
}  // CHECK-NEXT:   }

// CHECK-LABEL: "bef.invokable" ["opaque_round_trip"]
// CHECK-NEXT:    %0:2 = bef.opaque "simple.primitive" []() : () -> (!bef.unknown, !bef.unknown)
// CHECK-NEXT:    bef.opaque "other.primitive" [42 : si8](%0#1, %0#0) : (!bef.unknown, !bef.unknown) -> ()
// CHECK-NEXT:    bef.output
// CHECK-NEXT:  }
bef.invokable @opaque_round_trip() {
  %0:2 = "bef.opaque"() {attrs = [], opName = "simple.primitive"} : () -> (!bef.unknown, !bef.unknown)
  "bef.opaque"(%0#1, %0#0) {attrs = [42 : si8], opName = "other.primitive"} : (!bef.unknown, !bef.unknown) -> ()
  bef.output
}

// CHECK-LABEL: bef.opaque "some.container" ["multi_region"]() : () -> () {
// CHECK-NEXT:   bef.opaque "a.firstregion" []() : () -> ()
// CHECK-NEXT:   bef.output
// CHECK-NEXT: }, {
// CHECK-NEXT:   bef.opaque "a.secondregion" []() : () -> ()
// CHECK-NEXT:   bef.output
// CHECK-NEXT: }
"some.container"() ({
  "a.firstregion"() : () -> ()
}, {
  "a.secondregion"() : () -> ()
}) {sym_name = "multi_region"} : () -> ()

// CHECK-LABEL: "bef.invokable" ["controlflow_if"]
// CHECK-NEXT: ^bb0(%arg0: !bef.unknown, %arg1: !bef.unknown, %arg2: !bef.unknown):
// CHECK-NEXT:   %0 = bef.opaque "grt.if" nonstrict [](%arg0, %arg1, %arg2) : (!bef.unknown, !bef.unknown, !bef.unknown) -> !bef.unknown {
// CHECK-NEXT:   ^bb0(%arg3: !bef.unknown, %arg4: !bef.unknown):
// CHECK-NEXT:     bef.output %arg3 : !bef.unknown
// CHECK-NEXT:   }, {
// CHECK-NEXT:   ^bb0(%arg3: !bef.unknown, %arg4: !bef.unknown):
// CHECK-NEXT:     bef.output %arg4 : !bef.unknown
// CHECK-NEXT:   }
// CHECK-NEXT:   bef.output %0 : !bef.unknown
// CHECK-NEXT: }
bef.invokable @controlflow_if(%cond: i1, %v1: si32, %v2: si32) -> si32 {
  %res = grt.if %cond (%v1: si32, %v2: si32) -> (si32) {
    bef.output %v1 : si32
  } else {
    bef.output %v2 : si32
  }

  bef.output %res : si32
}

// CHECK-NEXT: }
