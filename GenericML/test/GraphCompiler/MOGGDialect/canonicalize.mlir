// RUN: mo-opt --canonicalize --split-input-file %s | FileCheck %s

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

mo.graph @genericRegionChainCreateNoInline(%arg0: index, %arg1: index) {
  %ch = grt.chain.create(%arg0 : index, %arg1 : index)

  mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mogg.generic.compiled.region (%ch) : (%in : !mef.chain) {
    mogg.output
  } { device = #M.device_ref<"cpu", 0>, inOutSignature = #M.inout_sig<"."> }

  mo.output
}

// CHECK-LABEL: mo.graph @genericRegionChainCreateNoInline
// CHECK:       (%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
// CHECK:       %[[CH:.*]] = grt.chain.create (%[[ARG0]]: index, %[[ARG1]]: index)
// CHECK:       mogg.generic.compiled.region (%[[CH]])
// CHECK:       inOutSignature = #M.inout_sig<".">
// CHECK:       mogg.generic.compiled.region (%[[CH]])
// CHECK:       inOutSignature = #M.inout_sig<".">

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
