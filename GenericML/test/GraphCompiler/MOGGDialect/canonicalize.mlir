// RUN: mo-opt --canonicalize --split-input-file %s | FileCheck %s

mo.graph @allNullInputFusionInterface(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2], f32>, %arg3: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.output %arg2 : !mo.tensor<[2], f32>
  } {inputFusionInterface = ["", ""]}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @allNullInputFusionInterface
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]], %[[ARG1:.*]])
// CHECK-NOT: inputFusionInterface

// -----

mo.graph @allNullOutputFusionInterface(%arg0: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0) : (%arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.output %arg1 : !mo.tensor<[2], f32>
  } {outputFusionInterface = [""]}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @allNullOutputFusionInterface
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]])
// CHECK-NOT: outputFusionInterface

// -----

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

// -----

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

// -----

mo.graph @partiallyNullInterface(%arg0: !mo.tensor<[2], f32>, %arg1: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
  %0 = mogg.kernel (%arg0, %arg1) : (%arg2: !mo.tensor<[2], f32>, %arg3: !mo.tensor<[2], f32>) -> !mo.tensor<[2], f32> {
    mogg.lambda @input_0_fn = (%arg4: !mogg.indices) -> !mogg.simd<f32>{
      %9 = mogg.call "simd_load"(%arg2, %arg4) : !mo.tensor<[2], f32>, !mogg.indices -> !mogg.simd<f32>
      mogg.output %9 : !mogg.simd<f32>
    }
    mogg.output %arg2 : !mo.tensor<[2], f32>
  } {inputFusionInterface = ["", "input_0_fn"]}
  mo.output %0 : !mo.tensor<[2], f32>
}

// CHECK: mo.graph @partiallyNullInterface
// CHECK-NEXT: mogg.kernel (%[[ARG0:.*]], %[[ARG1:.*]])
// CHECK: inputFusionInterface

// -----

mo.graph @genericRegionChainCreate(%arg0: index, %arg1: index) {
  %ch = grt.chain.create(%arg0 : index, %arg1 : index)

   mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionChainCreate
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       mogg.generic.compiled.region (%[[ARG0]], %[[ARG1]])
// CHECK:       inOutSignature = #M.inout_sig<"..">

// -----

mo.graph @genericRegionEmptyChain() {
  %ch = grt.chain.create()

   mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionEmptyChain
// CHECK:       mogg.generic.compiled.region ()
// CHECK:       inOutSignature = #M.inout_sig<"">
