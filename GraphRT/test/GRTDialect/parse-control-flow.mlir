// RUN: graphrt-opt -allow-unregistered-dialect %s | graphrt-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: bef.func @repeat(%arg0: i32, %arg1: f32) -> i32 {
bef.func @repeat(%arg0: i32, %arg1: f32) -> i32 {
  // CHECK: [[RES:%[0-9]+]]:2 = grt.repeat.i32 %arg0 (%arg0: i32, %arg1: f32) {
  %res1, %res2 = grt.repeat.i32 %arg0 (%arg0: i32, %arg1: f32) {
    // CHECK-NEXT: "use"(%arg0, %arg1)
    "use"(%arg0, %arg1) : (i32, f32) -> ()

    bef.output %arg0, %arg1 : i32, f32
  }

  // Zero argument loop.

  // CHECK: grt.repeat.i32 %arg0 () {
  grt.repeat.i32 %arg0 () {
    bef.output
  }

  // CHECK: bef.output [[RES]]#0
  bef.output %res1 : i32
}
