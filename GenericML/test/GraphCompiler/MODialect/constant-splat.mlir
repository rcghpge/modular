// RUN: mo-opt %s --cse --canonicalize --shape-inference --symbolic-optimize --cse --canonicalize | FileCheck %s

// This tests that splatted constants survive in their splatted form through
// our shape optimization pipeline.

// CHECK-LABEL: mo.graph @splat()
mo.graph @splat() -> !mo.tensor<[288], f32> {
    %0 = mo.constant {value = #M.dense_array<1.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
    %1 = mo.constant {value = #M.dense_array<288> : tensor<1xsi64>} : !mo.tensor<[1], si64>
    %2 = mo.broadcast_to(%0, %1) : (!mo.tensor<[], f32>, !mo.tensor<[1], si64>) -> !mo.tensor<[288], f32>

    // CHECK-NEXT: %[[SPLAT:.*]] = mo.constant {value = #M.dense_array<1.000000e+00> : !M.array<1xf32>} : !mo.tensor<[], f32>
    // CHECK-NEXT: %[[BCAST:.*]] = mo.static.broadcast_to(%[[SPLAT]]) : !mo.tensor<[], f32> -> !mo.tensor<[288], f32>
    // CHECK-NEXT: mo.output %[[BCAST]]
    mo.output %2 : !mo.tensor<[288], f32>
}
