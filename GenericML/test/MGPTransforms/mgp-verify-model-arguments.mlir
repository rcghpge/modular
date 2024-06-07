// RUN: mo-opt --verify-model-arguments %s | FileCheck %s

// CHECK-LABEL:  mgp.model @static
// CHECK:        } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.tensor<2x4xf32>, %[[ARG3:.*]]: !mgp.tensor<2x4xf32>) -> (!mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>) {
// CHECK-NEXT:     %[[EXTRACT1:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[ARG2]] shape_only true : !mgp.tensor<2x4xf32>
// CHECK-NEXT:     %[[EQUAL1:.*]] = mgp.tensor_spec.equal.static %[[EXTRACT1]] : !mgp.tensor_spec<2x4xunk>
// CHECK-NEXT:     %[[ASSERT1:.*]] = mgp.assert %[[EQUAL1:.*]]
// CHECK-NEXT:     %[[EXTRACT2:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[ARG3]] shape_only true : !mgp.tensor<2x4xf32>
// CHECK-NEXT:     %[[EQUAL2:.*]] = mgp.tensor_spec.equal.static %[[EXTRACT2]] : !mgp.tensor_spec<2x4xunk>
// CHECK-NEXT:     %[[ASSERT2:.*]] = mgp.assert %[[EQUAL2]]
// CHECK-NEXT:     mef.output %[[CH0]], %[[ARG2]], %[[ARG3]] : !mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>
// CHECK-NEXT:   }

mgp.model @static : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.tensor<2x4xf32>, %arg3: !mgp.tensor<2x4xf32>) -> (!mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.tensor<2x4xf32>, !mgp.tensor<2x4xf32>
}

// Expecting no check because all dimensions are unknown.
// CHECK-LABEL: mgp.model @dynamic
// CHECK:       } execute
// CHECK-NEXT:    mef.output
mgp.model @dynamic : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.tensor<?x?xf32>, %arg3: !mgp.tensor<?x?xf32>) -> (!mef.chain, !mgp.tensor<?x?xf32>, !mgp.tensor<?x?xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.tensor<?x?xf32>, !mgp.tensor<?x?xf32>
}

// CHECK-LABEL:  mgp.model @mixed
// CHECK:        } execute (%[[CH0:.*]]: !mef.chain, %[[CTX:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.tensor<?x4xf32>, %[[ARG3:.*]]: !mgp.tensor<?x4xf32>) -> (!mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>) {
// CHECK-NEXT:     %[[EXTRACT1:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[ARG2]] shape_only true : !mgp.tensor<?x4xf32>
// CHECK-NEXT:     %[[EQUAL1:.*]] = mgp.tensor_spec.equal.static %[[EXTRACT1]] : !mgp.tensor_spec<?x4xunk>
// CHECK-NEXT:     %[[ASSERT1:.*]] = mgp.assert %[[EQUAL1:.*]]
// CHECK-NEXT:     %[[EXTRACT2:.*]] = mgp.tensor.extract.tensor_spec %[[CTX]][1], %[[ARG3]] shape_only true : !mgp.tensor<?x4xf32>
// CHECK-NEXT:     %[[EQUAL2:.*]] = mgp.tensor_spec.equal.static %[[EXTRACT2]] : !mgp.tensor_spec<?x4xunk>
// CHECK-NEXT:     %[[ASSERT2:.*]] = mgp.assert %[[EQUAL2]]
// CHECK-NEXT:     mef.output %[[CH0]], %[[ARG2]], %[[ARG3]] : !mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>
// CHECK-NEXT:   }
mgp.model @mixed : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 2
  %ch1 = mgp.check_runtime %ch0, %ctx[0], <host=<"cpu", 0>, devices=[<ref=<"cpu", 0>, target=<triple="", arch="", features="">>]>
  %ch2 = mgp.runtime.create %ch1, %ctx[1]
  mef.output %ch2, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.tensor<?x4xf32>, %arg3: !mgp.tensor<?x4xf32>) -> (!mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.tensor<?x4xf32>, !mgp.tensor<?x4xf32>
}
