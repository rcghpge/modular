// RUN: graphrt-opt -allow-unregistered-dialect %s | graphrt-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: bef.invokable @if(%arg0: i1, %arg1: si32, %arg2: si32) -> si32 {

bef.invokable @if(%cond: i1, %v1: si32, %v2: si32) -> si32 {
  // CHECK: [[RES:%[0-9]+]] = grt.if %arg0 (%arg1: si32, %arg2: si32) -> si32 {
  %res = grt.if %cond (%v1: si32, %v2: si32) -> si32 {
    // CHECK-NEXT: bef.output %arg1
    bef.output %v1 : si32
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: bef.output %arg2
    bef.output %v2 : si32
  }


  grt.if %cond () -> () {   // CHECK: grt.if %arg0 () -> () {
    bef.output              // CHECK-NEXT: bef.output
  } else {                  // CHECK-NEXT: } else {
    bef.output              // CHECK-NEXT: bef.output
  }                         // CHECK-NEXT: }

  // CHECK: bef.output [[RES]]
  bef.output %res : si32
}

// CHECK-LABEL: bef.invokable @repeat(%arg0: si32, %arg1: f32) -> si32 {
bef.invokable @repeat(%arg0: si32, %arg1: f32) -> si32 {
  // CHECK: [[RES:%[0-9]+]]:2 = grt.repeat.i32 %arg0 (%arg0: si32, %arg1: f32) {
  %res1, %res2 = grt.repeat.i32 %arg0 (%arg0: si32, %arg1: f32) {
    // CHECK-NEXT: "use"(%arg0, %arg1)
    "use"(%arg0, %arg1) : (si32, f32) -> ()

    bef.output %arg0, %arg1 : si32, f32
  }

  // Zero argument loop.

  // CHECK: grt.repeat.i32 %arg0 () {
  grt.repeat.i32 %arg0 () {
    bef.output
  }

  // CHECK: bef.output [[RES]]#0
  bef.output %res1 : si32
}
