// RUN: bef-translate -export-bef -allow-unregistered-dialect %s > %t.bef
// RUN: bef-executor %t.bef | FileCheck %s

bef.func @fib(%arg: i32) -> i32 {
  %one = grt.constant.i32 1
  %cond = grt.lessequal.i32 %arg, %one
  %res = grt.if %cond (%arg: i32) -> i32 {
    bef.output %arg : i32
  } else {
    %min_one = grt.constant.i32 -1
    %n_min_one = grt.add.i32 %arg, %min_one
    %n_min_two = grt.add.i32 %n_min_one, %min_one
    %fib_n_min_one = grt.call @fib(%n_min_one) : (i32) -> (i32)
    %fib_n_min_two = grt.call @fib(%n_min_two) : (i32) -> (i32)
    %sum = grt.add.i32 %fib_n_min_one, %fib_n_min_two
    bef.output %sum : i32
  }

  bef.output %res : i32
}

// CHECK-LABEL: --- Running 'fib_driver'
bef.func @fib_driver() {
  %zero = grt.constant.i32 0
  %fib0 = grt.call @fib(%zero) : (i32) -> (i32)

  %four = grt.constant.i32 4
  %fib4 = grt.call @fib(%four) : (i32) -> (i32)

  %eight = grt.constant.i32 8
  %fib8 = grt.call @fib(%eight) : (i32) -> (i32)

  %sixteen = grt.constant.i32 16
  %fib16 = grt.call @fib(%sixteen) : (i32) -> (i32)

  %ch0 = grt.chain.new

  // CHECK: int32 = 0
  %ch1 = grt.print.i32 %fib0, %ch0

  // CHECK: int32 = 3
  %ch2 = grt.print.i32 %fib4, %ch1

  // CHECK: int32 = 21
  %ch3 = grt.print.i32 %fib8, %ch2

  // CHECK: int32 = 987
  %ch4 = grt.print.i32 %fib16, %ch3

  bef.output
}

