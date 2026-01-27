// RUN: not mo-opt --mo-legalize-rmo-operators %s 2>&1 | FileCheck %s

mo.graph @forward() -> !mo.tensor<[2, 3], si32> {
  %data = mo.constant {value = #M.dense_array<10, 20, 30, 40, 50, 60> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %updates = mo.constant {value = #M.dense_array<100, 200, 300, 1000, 2000, 3000> : tensor<6xsi32>} : !mo.tensor<[6], si32>
  %indices = mo.constant {value = #M.dense_array<1, 0> : tensor<2xsi32>} : !mo.tensor<[2, 1], si32>

  // NOTE: we do not use --verify-diagnostics as other diagnostics are emitted
  // and we seem to need to check all of them.
  //
  // CHECK: error: [LegalizeRMOOps] RMO rebinding failed for op in operand index #1 original and expected types are not compatible.
  // CHECK-NEXT:     original type is     : '!mo.tensor<[6], si32>'
  // CHECK-NEXT:     RMO expected type is : '!mo.tensor<[RMO_D1, RMO_D2], si32>'
  %0 = rmo.mo.scatter_nd(%data, %updates, %indices) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[6], si32>, !mo.tensor<[2, 1], si32>) -> !mo.tensor<[2, 3], si32>
  mo.output %0 : !mo.tensor<[2, 3], si32>
}
