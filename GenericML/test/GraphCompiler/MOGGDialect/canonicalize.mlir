// RUN: mo-opt --canonicalize %s | FileCheck %s

mo.graph @allNullInputFusionInterface(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2], f32>, %arg3: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.output %arg2 : !mo.tensor<[2], f32>
  } {inputFusionInterface = ["", ""]}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @allNullInputFusionInterface
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]], %[[ARG1:.*]])
// CHECK-NOT: inputFusionInterface

mo.graph @allNullOutputFusionInterface(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0) : (%arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.output %arg1 : !mo.tensor<[2], f32>
  } {outputFusionInterface = [""]}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @allNullOutputFusionInterface
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]])
// CHECK-NOT: outputFusionInterface

mo.graph @allNullFusionInterfaces(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2], f32>, %arg3: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.output %arg2 : !mo.tensor<[2], f32>
  } {inputFusionInterface = ["", ""], outputFusionInterface = [""]}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @allNullFusionInterfaces
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]], %[[ARG1:.*]])
// CHECK-NOT: inputFusionInterface
// CHECK-NOT: outputFusionInterface

mo.graph @noInterfaces(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0) : (%arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.output %arg1 : !mo.tensor<[2], f32>
  } {}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @noInterfaces
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]])
// CHECK-NOT: inputFusionInterface
// CHECK-NOT: outputFusionInterface

mo.graph @partiallyNullInterface(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2], f32>, %arg3: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.output %arg2 : !mo.tensor<[2], f32>
  } {inputFusionInterface = ["", "foo"]}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @partiallyNullInterface
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]], %[[ARG1:.*]])
// CHECK: inputFusionInterface
