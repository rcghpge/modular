// RUN: mgp-opt --verify-model-arguments %s | FileCheck %s

// CHECK-LABEL:  mgp.model @static
// CHECK:        } execute (%[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<2x4xf32>, %[[ARG3:.*]]: !mgp.cpu.tensor<2x4xf32>) -> (!mef.chain, !mgp.cpu.tensor<2x4xf32>, !mgp.cpu.tensor<2x4xf32>) {
// CHECK-NEXT:     %[[EXTRACT1:.*]] = mgp.cpu.tensor.extract.shape %[[ARG2]] : !mgp.cpu.tensor<2x4xf32>
// CHECK-NEXT:     %[[EQUAL1:.*]] = mgp.cpu.tensor_spec.equal.static %[[EXTRACT1]] : !mgp.tensor_spec<2x4xunk>
// CHECK-NEXT:     %[[ASSERT1:.*]] = mgp.assert %[[EQUAL1:.*]]
// CHECK-NEXT:     %[[EXTRACT2:.*]] = mgp.cpu.tensor.extract.shape %[[ARG3]] : !mgp.cpu.tensor<2x4xf32>
// CHECK-NEXT:     %[[EQUAL2:.*]] = mgp.cpu.tensor_spec.equal.static %[[EXTRACT2]] : !mgp.tensor_spec<2x4xunk>
// CHECK-NEXT:     %[[ASSERT2:.*]] = mgp.assert %[[EQUAL2]]
// CHECK-NEXT:     mef.output %[[ARG0]], %[[ARG2]], %[[ARG3]] : !mef.chain, !mgp.cpu.tensor<2x4xf32>, !mgp.cpu.tensor<2x4xf32>
// CHECK-NEXT:   }

mgp.model @static : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<2x4xf32>, %arg3: !mgp.cpu.tensor<2x4xf32>) -> (!mef.chain, !mgp.cpu.tensor<2x4xf32>, !mgp.cpu.tensor<2x4xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.cpu.tensor<2x4xf32>, !mgp.cpu.tensor<2x4xf32>
}

// Expecting no check because all dimensions are unknown.
// CHECK-LABEL: mgp.model @dynamic
// CHECK:       } execute
// CHECK-NEXT:    mef.output
mgp.model @dynamic : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x?xf32>, %arg3: !mgp.cpu.tensor<?x?xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x?xf32>, !mgp.cpu.tensor<?x?xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.cpu.tensor<?x?xf32>, !mgp.cpu.tensor<?x?xf32>
}

// CHECK-LABEL:  mgp.model @mixed
// CHECK:        } execute (%[[ARG0:.*]]: !mef.chain, %[[ARG1:.*]]: !mgp.context, %[[ARG2:.*]]: !mgp.cpu.tensor<?x4xf32>, %[[ARG3:.*]]: !mgp.cpu.tensor<?x4xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x4xf32>, !mgp.cpu.tensor<?x4xf32>) {
// CHECK-NEXT:     %[[EXTRACT1:.*]] = mgp.cpu.tensor.extract.shape %[[ARG2]] : !mgp.cpu.tensor<?x4xf32>
// CHECK-NEXT:     %[[EQUAL1:.*]] = mgp.cpu.tensor_spec.equal.static %[[EXTRACT1]] : !mgp.tensor_spec<?x4xunk>
// CHECK-NEXT:     %[[ASSERT1:.*]] = mgp.assert %[[EQUAL1:.*]]
// CHECK-NEXT:     %[[EXTRACT2:.*]] = mgp.cpu.tensor.extract.shape %[[ARG3]] : !mgp.cpu.tensor<?x4xf32>
// CHECK-NEXT:     %[[EQUAL2:.*]] = mgp.cpu.tensor_spec.equal.static %[[EXTRACT2]] : !mgp.tensor_spec<?x4xunk>
// CHECK-NEXT:     %[[ASSERT2:.*]] = mgp.assert %[[EQUAL2]]
// CHECK-NEXT:     mef.output %[[ARG0]], %[[ARG2]], %[[ARG3]] : !mef.chain, !mgp.cpu.tensor<?x4xf32>, !mgp.cpu.tensor<?x4xf32>
// CHECK-NEXT:   }
mgp.model @mixed : mof init {
  %ch0 = grt.chain.create ()
  %ctx = mgp.context.create 1
  mef.output %ch0, %ctx : !mef.chain, !mgp.context
} execute (%arg0: !mef.chain, %arg1: !mgp.context, %arg2: !mgp.cpu.tensor<?x4xf32>, %arg3: !mgp.cpu.tensor<?x4xf32>) -> (!mef.chain, !mgp.cpu.tensor<?x4xf32>, !mgp.cpu.tensor<?x4xf32>) {
  mef.output %arg0, %arg2, %arg3 : !mef.chain, !mgp.cpu.tensor<?x4xf32>, !mgp.cpu.tensor<?x4xf32>
}
