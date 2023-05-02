// RUN: graphrt-opt -allow-unregistered-dialect %s | graphrt-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: mef.invokable @if
// CHECK-SAME: %[[ARG0:.*]]: i1, %[[ARG1:.*]]: si32, %[[ARG2:.*]]: si32
mef.invokable @if(%cond: i1, %v1: si32, %v2: si32) -> si32 {
  // CHECK: %[[RES:.*]] = grt.if %[[ARG0]] (%[[ARG1]]: si32, %[[ARG2]]: si32) -> si32 {
  %res = grt.if %cond (%v1: si32, %v2: si32) -> si32 {
    // CHECK-NEXT: mef.output %[[ARG1]]
    mef.output %v1 : si32
    // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: mef.output %[[ARG2]]
    mef.output %v2 : si32
  }

  // CHECK: grt.if %[[ARG0]] () -> () {
  grt.if %cond () -> () {
    // CHECK-NEXT: mef.output
    mef.output
  // CHECK-NEXT: } else {
  } else {
    // CHECK-NEXT: mef.output
    mef.output
  // CHECK-NEXT: }
  }

  // CHECK: mef.output %[[RES]]
  mef.output %res : si32
}

// CHECK-LABEL: mef.invokable @while
// CHECK-SAME: %[[ARG1:.*]]: i32, %[[ARG2:.*]]: f32
// CHECK: %[[RES:.*]]:2 = grt.while (%[[ARG1]]: i32, %[[ARG2]]: f32) {
// CHECK-NEXT: %[[COND:.*]] = "my.cond"(%[[ARG1]], %[[ARG2]])
// CHECK-NEXT: mef.output %[[ARG1]], %[[ARG2]], %[[COND]]
// CHECK-NEXT: } do {
// CHECK-NEXT: mef.output %[[ARG1]], %[[ARG2]]
// CHECK-NEXT: }
// CHECK-NEXT: mef.output %[[RES]]#0, %[[RES]]#1
mef.invokable @while(%v1: i32, %v2: f32) -> (i32, f32) {
  %res1, %res2 = grt.while (%v1: i32, %v2: f32) {
    %c = "my.cond"(%v1, %v2) : (i32, f32) -> i1
    mef.output %v1, %v2, %c : i32, f32, i1
  } do {
    mef.output %v1, %v2 : i32, f32
  }
  mef.output %res1, %res2 : i32, f32
}

// CHECK-LABEL: mef.invokable @repeat
// CHECK-SAME: %[[ARG0:.*]]: si32, %[[ARG1:.*]]: f32
mef.invokable @repeat(%arg0: si32, %arg1: f32) -> si32 {
  // CHECK: %[[RES:.*]]:2 = grt.repeat.i32 %[[ARG0]] (%[[ARG0]]: si32, %[[ARG1]]: f32) {
  %res1, %res2 = grt.repeat.i32 %arg0 (%arg0: si32, %arg1: f32) {
    // CHECK-NEXT: "my.use"(%[[ARG0]], %[[ARG1]])
    "my.use"(%arg0, %arg1) : (si32, f32) -> ()

    mef.output %arg0, %arg1 : si32, f32
  }

  // Zero argument loop.

  // CHECK: grt.repeat.i32 %[[ARG0]] () {
  grt.repeat.i32 %arg0 () {
    mef.output
  }

  // CHECK: mef.output %[[RES]]#0
  mef.output %res1 : si32
}
