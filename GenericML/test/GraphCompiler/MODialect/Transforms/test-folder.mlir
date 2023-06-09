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

mo.graph @fold_cast(%arg0 : !mo.tensor<[100, ?, 50], f32>) -> (!mo.tensor<[100, ?, 50], f32>) {
    %output = mo.cast(%arg0) : (!mo.tensor<[100, ?, 50], f32>) -> !mo.tensor<[100, ?, 50],f32>
    // CHECK-LABEL: mo.graph @fold_cast
    // CHECK: (%[[ARG0:.*]]: !mo.tensor<[100, ?, 50], f32>)
    // CHECK-NEXT: mo.output %[[ARG0]]
    mo.output %output : !mo.tensor<[100, ?, 50], f32>
}

mo.graph @fold_div() -> (!mo.tensor<[7], si32>) {
    %0 = "mo.constant"() {value = #M.dense_array<1, -5, 10, 3, 1, 250, -343> : tensor<7xsi32>} : () -> !mo.tensor<[7], si32>
    %1 = "mo.constant"() {value = #M.dense_array<1, 5, -2, 2, 3, 5, -343> : tensor<7xsi32>} : () -> !mo.tensor<[7], si32>

    %output = mo.div(%0, %1) : !mo.tensor<[7], si32>
    // CHECK-LABEL: mo.graph @fold_div
    // CHECK-NEXT: %[[CST:.*]] = mo.constant {value = #M.dense_array<1, -1, -5, 1, 0, 50, 1> : tensor<7xsi32>}
    // CHECK-NEXT: mo.output %[[CST]]
    mo.output %output : !mo.tensor<[7], si32>
}


mo.graph @fold_add0_f64_lhs(%arg0: !mo.tensor<[5, 500, 100], f64>) -> !mo.tensor<[5, 500, 100], f64> {
    %0 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<1x1xf64>} : !mo.tensor<[1, 1], f64>
    %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1, 1], f64> -> !mo.tensor<[5, 500, 100], f64>
    %2 = mo.add(%1, %arg0) : !mo.tensor<[5, 500, 100], f64>
    mo.output %2 : !mo.tensor<[5, 500, 100], f64>
}
// CHECK-LABEL: mo.graph @fold_add0_f64_lhs
// CHECK:(%[[ARG0:.*]]: !mo.tensor<
// CHECK-NEXT: mo.output %[[ARG0]]

mo.graph @fold_add0_f64_rhs(%arg0: !mo.tensor<[5, 500, 100], f64>) -> !mo.tensor<[5, 500, 100], f64> {
    %0 = mo.constant {value = #M.dense_array<0.000000e+00> : tensor<1x1xf64>} : !mo.tensor<[1, 1], f64>
    %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1, 1], f64> -> !mo.tensor<[5, 500, 100], f64>
    %2 = mo.add(%arg0, %1) : !mo.tensor<[5, 500, 100], f64>
    mo.output %2 : !mo.tensor<[5, 500, 100], f64>
}
// CHECK-LABEL: mo.graph @fold_add0_f64_rhs
// CHECK:(%[[ARG0:.*]]: !mo.tensor<
// CHECK-NEXT: mo.output %[[ARG0]]


mo.graph @fold_add0_si32(%arg0: !mo.tensor<[5, 500, 100], si32>) -> !mo.tensor<[5, 500, 100], si32> {
    %0 = mo.constant {value = #M.dense_array<0> : tensor<1x1xsi32>} : !mo.tensor<[1, 1], si32>
    %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1, 1], si32> -> !mo.tensor<[5, 500, 100], si32>
    %2 = mo.add(%arg0, %1) : !mo.tensor<[5, 500, 100], si32>
    mo.output %2 : !mo.tensor<[5, 500, 100], si32>
}
// CHECK-LABEL: mo.graph @fold_add0_si32
// CHECK:(%[[ARG0:.*]]: !mo.tensor<
// CHECK-NEXT: mo.output %[[ARG0]]


mo.graph @fold_add0_ui64(%arg0: !mo.tensor<[5, 500, 100], ui64>) -> !mo.tensor<[5, 500, 100], ui64> {
    %0 = mo.constant {value = #M.dense_array<0> : tensor<1x1xui64>} : !mo.tensor<[1, 1], ui64>
    %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1, 1], ui64> -> !mo.tensor<[5, 500, 100], ui64>
    %2 = mo.add(%arg0, %1) : !mo.tensor<[5, 500, 100], ui64>
    mo.output %2 : !mo.tensor<[5, 500, 100], ui64>
}
// CHECK-LABEL: mo.graph @fold_add0_ui64
// CHECK:(%[[ARG0:.*]]: !mo.tensor<
// CHECK-NEXT: mo.output %[[ARG0]]


mo.graph @fold_add0_no_fold_ui64(%arg0: !mo.tensor<[5, 500, 100], ui64>) -> !mo.tensor<[5, 500, 100], ui64> {
    %0 = mo.constant {value = #M.dense_array<1> : tensor<1x1xui64>} : !mo.tensor<[1, 1], ui64>
    %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1, 1], ui64> -> !mo.tensor<[5, 500, 100], ui64>
    %2 = mo.add(%arg0, %1) : !mo.tensor<[5, 500, 100], ui64>
    mo.output %2 : !mo.tensor<[5, 500, 100], ui64>
}
// CHECK-LABEL: mo.graph @fold_add0_no_fold_ui64
// CHECK:(%[[ARG0:.*]]: !mo.tensor<
// CHECK-NEXT: mo.constant
// CHECK-NEXT: mo.static.broadcast_to
// CHECK-NEXT: %[[OUT:.*]] = mo.add

mo.graph @fold_add0_no_fold_f64(%arg0: !mo.tensor<[5, 500, 100], f64>) -> !mo.tensor<[5, 500, 100], f64> {
    %0 = mo.constant {value = #M.dense_array<0.000001e+00> : tensor<1x1xf64>} : !mo.tensor<[1, 1], f64>
    %1 = mo.static.broadcast_to(%0) : !mo.tensor<[1, 1], f64> -> !mo.tensor<[5, 500, 100], f64>
    %2 = mo.add(%arg0, %1) : !mo.tensor<[5, 500, 100], f64>
    mo.output %2 : !mo.tensor<[5, 500, 100], f64>
}
// CHECK-LABEL: mo.graph @fold_add0_no_fold_f64
// CHECK:(%[[ARG0:.*]]: !mo.tensor<
// CHECK-NEXT: mo.constant
// CHECK-NEXT: mo.static.broadcast_to
// CHECK-NEXT: %[[OUT:.*]] = mo.add

mo.graph @fold_slice(%arg0: !mo.tensor<[5, 500], f64>, 
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64> {
    %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[5, 500], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64>
    mo.output %2 : !mo.tensor<[5, 500], f64>
}
// CHECK-LABEL: mo.graph @fold_slice
// CHECK:(%[[ARG0:.*]]: !mo.tensor<[5, 500
// CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[5, 500], f64>


// There is some combination which should be able to produce the equiv of:
// np.arange(0, 15).reshape(3, 5)[::-1, ::-1]
// Which is the reverse with the same shape.
mo.graph @no_fold_slice_reverse(%arg0: !mo.tensor<[5, 500], f64>, 
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64> {
    %step = mo.constant {value = #M.dense_array<-1, -1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[5, 500], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 500], f64>
    mo.output %2 : !mo.tensor<[5, 500], f64>
}
// CHECK-LABEL: mo.graph @no_fold_slice_reverse
// CHECK:(%[[ARG0:.*]]: !mo.tensor<[5, 500
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.slice
// CHECK-NEXT: mo.output %[[OUT]]


mo.graph @no_fold_slice_unknown(%arg0: !mo.tensor<[5, ?], f64>, 
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[5, ?], f64> {
    %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[5, ?], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, ?], f64>
    mo.output %2 : !mo.tensor<[5, ?], f64>
}
// CHECK-LABEL: mo.graph @no_fold_slice_unknown
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.slice
// CHECK-NEXT: mo.output %[[OUT]]

mo.graph @no_fold_slice_shape_mismatch(%arg0: !mo.tensor<[5, 500], f64>, 
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[5, 250], f64> {
    %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[5, 500], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[5, 250], f64>
    mo.output %2 : !mo.tensor<[5, 250], f64>
}
// CHECK-LABEL: mo.graph @no_fold_slice_shape_mismatch
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.slice
// CHECK-NEXT: mo.output %[[OUT]]


mo.graph @fold_slice_parameter<N, M>(%arg0: !mo.tensor<[N, M], f64>, 
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[N, M], f64> {
    %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[N, M], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[N, M], f64>
    mo.output %2 : !mo.tensor<[N, M], f64>
}
// CHECK-LABEL: mo.graph @fold_slice_parameter<N, M>
// CHECK:(%[[ARG0:.*]]: !mo.tensor<[N, M
// CHECK-NEXT: mo.output %[[ARG0]] : !mo.tensor<[N, M], f64>


mo.graph @no_fold_slice_parameter<N, M>(%arg0: !mo.tensor<[N, M], f64>, 
    %start: !mo.tensor<[2], si64>,
    %stop: !mo.tensor<[2], si64>) -> !mo.tensor<[M, N], f64> {
    %step = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
    %2 = mo.slice(%arg0, %start, %stop, %step) : (!mo.tensor<[N, M], f64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[M, N], f64>
    mo.output %2 : !mo.tensor<[M, N], f64>
}
// CHECK-LABEL: mo.graph @no_fold_slice_parameter<N, M>
// CHECK-NEXT: mo.constant
// CHECK-NEXT: %[[OUT:.*]] = mo.slice
// CHECK-NEXT: mo.output %[[OUT]]
