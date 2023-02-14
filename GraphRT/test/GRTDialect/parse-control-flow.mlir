// RUN: graphrt-opt -allow-unregistered-dialect %s | graphrt-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mef.invokable @if
// CHECK-SAME: %[[ARG0:.*]]: i1, %[[ARG1:.*]]: si32, %[[ARG2:.*]]: si32

mef.invokable @if(%cond: i1, %v1: si32, %v2: si32) -> si32 {
  // CHECK: [[RES:%[0-9]+]] = grt.if %[[ARG0]] (%[[ARG1]]: si32, %[[ARG2]]: si32) -> si32 {
  %res = grt.if %cond (%v1: si32, %v2: si32) -> si32 {
    // CHECK-NEXT: mef.output %[[ARG1]]
    mef.output %v1 : si32
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: mef.output %[[ARG2]]
    mef.output %v2 : si32
  }


  grt.if %cond () -> () {   // CHECK: grt.if %[[ARG0]] () -> () {
    mef.output              // CHECK-NEXT: mef.output
  } else {                  // CHECK-NEXT: } else {
    mef.output              // CHECK-NEXT: mef.output
  }                         // CHECK-NEXT: }

  // CHECK: mef.output [[RES]]
  mef.output %res : si32
}

// CHECK-LABEL: mef.invokable @repeat
// CHECK-SAME: %[[ARG0:.*]]: si32, %[[ARG1:.*]]: f32
mef.invokable @repeat(%arg0: si32, %arg1: f32) -> si32 {
  // CHECK: [[RES:%[0-9]+]]:2 = grt.repeat.i32 %[[ARG0]] (%[[ARG0]]: si32, %[[ARG1]]: f32) {
  %res1, %res2 = grt.repeat.i32 %arg0 (%arg0: si32, %arg1: f32) {
    // CHECK-NEXT: "use"(%[[ARG0]], %[[ARG1]])
    "use"(%arg0, %arg1) : (si32, f32) -> ()

    mef.output %arg0, %arg1 : si32, f32
  }

  // Zero argument loop.

  // CHECK: grt.repeat.i32 %[[ARG0]] () {
  grt.repeat.i32 %arg0 () {
    mef.output
  }

  // CHECK: mef.output [[RES]]#0
  mef.output %res1 : si32
}
