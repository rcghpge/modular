// RUN: mo-opt %s --canonicalize --cse | FileCheck %s

// Test the folders for ops.

mo.graph @fold_split_scalar() -> (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>) {
    %axis = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
    %num_splits = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
    %v1 = mo.constant {value = #M.dense_array<0.0> : tensor<1xf32>} : !mo.tensor<[1], f32>
    %v2 = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
    %v3 = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>

    %o1 = mo.split[%axis : !mo.tensor<[], si32>] (%v1, %num_splits) : (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], f32>)
    %o2 = mo.split[%axis : !mo.tensor<[], si32>] (%v2, %num_splits) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si64>)
    %o3 = mo.split[%axis : !mo.tensor<[], si32>] (%v3, %num_splits) : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si32>)

    // CHECK-LABEL: mo.graph @fold_split_scalar
    // CHECK: %[[CSE0:.*]] = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<1xf32>} : !mo.tensor<[1], f32>
    // CHECK-NEXT: %[[CSE1:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[1], si64>
    // CHECK-NEXT: %[[CSE2:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi32>} : !mo.tensor<[1], si32>
    // CHECK-NEXT: mo.output %[[CSE0]], %[[CSE1]], %[[CSE2]] : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
    mo.output %o1, %o2, %o3 : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
}

mo.graph @fold_split_scalar_non_consts(%arg0 : !mo.tensor<[1], f32>, %arg1 : !mo.tensor<[1], si64>, %arg2 : !mo.tensor<[1], si32>) -> (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>) {
    %axis = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
    %num_splits = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[1], si64>

    %o1 = mo.split[%axis : !mo.tensor<[], si32>] (%arg0, %num_splits) : (!mo.tensor<[1], f32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], f32>)
    %o2 = mo.split[%axis : !mo.tensor<[], si32>] (%arg1, %num_splits) : (!mo.tensor<[1], si64>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si64>)
    %o3 = mo.split[%axis : !mo.tensor<[], si32>] (%arg2, %num_splits) : (!mo.tensor<[1], si32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[1], si32>)

    // CHECK-LABEL: mo.graph @fold_split_scalar_non_consts
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[1], f32>, %[[ARG1:.*]]: !mo.tensor<[1], si64>, %[[ARG2:.*]]: !mo.tensor<[1], si32>)
    // CHECK-NEXT: mo.output %[[ARG0]], %[[ARG1]], %[[ARG2]] : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
    mo.output %o1, %o2, %o3 : !mo.tensor<[1], f32>, !mo.tensor<[1], si64>, !mo.tensor<[1], si32>
}


mo.graph @fold_num_splits_1(%arg0 : !mo.tensor<[100, ?, 50], f32>) -> (!mo.tensor<[100, ?, 50], f32>) {
    %axis = mo.constant {value = #M.dense_array<0> : !M.array<1xsi32>} : !mo.tensor<[], si32>
    %num_splits = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[1], si64>
    %o1 = mo.split[%axis : !mo.tensor<[], si32>] (%arg0, %num_splits) : (!mo.tensor<[100, ?, 50], f32>, !mo.tensor<[1], si64>) -> (!mo.tensor<[100, ?, 50], f32>)

    // CHECK-LABEL: mo.graph @fold_num_splits_1
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[100, ?, 50], f32>)
    // CHECK-NEXT: mo.output %[[ARG0]]
    mo.output %o1 : !mo.tensor<[100, ?, 50], f32>
}
