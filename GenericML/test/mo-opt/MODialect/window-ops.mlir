// RUN: mo-opt -cse %s | mo-opt | FileCheck %s

// NOTE: Keep the tests grouped by op, and the order of ops alphabetical.

//===----------------------------------------------------------------------===//
// ConvOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @conv2d(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 5, 5, 16], f32>,
// CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 2, 2, 32], f32> {
mo.graph @conv2d(
  %input: !mo.tensor<[10, 5, 5, 16], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[10, 2, 2, 32], f32> {
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[NG:.*]] = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %ng = mo.constant {value = #M.dense_array<1> : tensor<si64>} : !mo.tensor<[], si64>
  // CHECK: mo.conv(%[[INPUT]], %[[FILTER]]) {filter_layout = ""} [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]], num_groups = %[[NG]]]
  %0 = mo.conv(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<[10, 5, 5, 16], f32>, !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<[10, 2, 2, 32], f32>
  mo.output %0 : !mo.tensor<[10, 2, 2, 32], f32>
}

// CHECK-LABEL: mo.graph @conv_dyn_rank
// CHECK-SAME: <[[SH0:[^,>]*]]: !mosh.ape, [[SH1:[^,>]*]]: !mosh.ape, [[SH2:[^,>]*]]: !mosh.ape, [[D0:[^,>]*]], [[D1:[^,>]*]], [[SH3:[^,>]*]]: !mosh.ape>(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[SH0]], f32>, %[[FILTER:.*]]: !mo.tensor<[[SH1]], f32>,
// CHECK-SAME: %[[ST:.*]]: !mo.tensor<[[SH2]], si64>, %[[DI:.*]]: !mo.tensor<{{\[}}[[D0]]{{\]}}, si64>, %[[PA:.*]]: !mo.tensor<{{\[}}[[D1]]{{\]}}, si64>,
// CHECK-SAME: %[[NG:.*]]: !mo.tensor<[], si64>) -> !mo.tensor<[[SH3]], f32> {
mo.graph @conv_dyn_rank<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, D0, D1, Sh3: !mosh.ape>(
  %input: !mo.tensor<Sh0, f32>,
  %filter: !mo.tensor<Sh1, f32>,
  %st: !mo.tensor<Sh2, si64>,
  %di: !mo.tensor<[D0], si64>,
  %pa: !mo.tensor<[D1], si64>,
  %ng: !mo.tensor<[], si64>
) -> !mo.tensor<Sh3, f32> {
  // CHECK: mo.conv(%[[INPUT]], %[[FILTER]]) {filter_layout = ""} [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]], num_groups = %[[NG]]]
  %0 = mo.conv(%input, %filter) {filter_layout = ""} [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>,
    !mo.tensor<Sh2, si64>, !mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>, !mo.tensor<[], si64>
  ) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

//===----------------------------------------------------------------------===//
// ConvTransposeOp
//===----------------------------------------------------------------------===//

// CHECK: mo.graph @conv_transpose2d(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[10, 2, 2, 32], f32>,
// CHECK-SAME: %[[FILTER:.*]]: !mo.tensor<[3, 3, 16, 32], f32>) -> !mo.tensor<[10, 4, 4, 16], f32> {
mo.graph @conv_transpose2d(
  %input: !mo.tensor<[10, 2, 2, 32], f32>,
  %filter: !mo.tensor<[3, 3, 16, 32], f32>
) -> !mo.tensor<[10, 4, 4, 16], f32> {
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[OP:.*]] = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %op = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: mo.conv_transpose(%[[INPUT]], %[[FILTER]])
  // CHECK-SAME: [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]], output_paddings = %[[OP]]]
  %0 = mo.conv_transpose(%input, %filter) {filter_layout = ""}
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<[10, 2, 2, 32], f32>, !mo.tensor<[3, 3, 16, 32], f32>,
    !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>
  ) -> !mo.tensor<[10, 4, 4, 16], f32>
  mo.output %0 : !mo.tensor<[10, 4, 4, 16], f32>
}

// CHECK: mo.graph @conv_transpose_dyn_rank<[[SH0:[^,>]*]]: !mosh.ape, [[SH1:[^,>]*]]: !mosh.ape, [[SH2:[^,>]*]]: !mosh.ape, [[D0:[^,>]*]], [[D1:[^,>]*]], [[SH3:[^,>]*]]: !mosh.ape>(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[SH0]], f32>, %[[FILTER:.*]]: !mo.tensor<[[SH1]], f32>,
// CHECK-SAME: %[[ST:.*]]: !mo.tensor<[[SH2]], si64>,
// CHECK-SAME: %[[DI:.*]]: !mo.tensor<{{\[}}[[D0]]{{\]}}, si64>, %[[PA:.*]]: !mo.tensor<{{\[}}[[D1]]{{\]}}, si64>, %[[OP:.*]]: !mo.tensor<{{\[}}[[D0]]{{\]}}, si64>
// CHECK-SAME:) -> !mo.tensor<[[SH3]], f32> {
mo.graph @conv_transpose_dyn_rank<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, D0, D1, Sh3: !mosh.ape>(
  %input: !mo.tensor<Sh0, f32>,
  %filter: !mo.tensor<Sh1, f32>,
  %st: !mo.tensor<Sh2, si64>,
  %di: !mo.tensor<[D0], si64>,
  %pa: !mo.tensor<[D1], si64>,
  %op: !mo.tensor<[D0], si64>
) -> !mo.tensor<Sh3, f32> {
  // CHECK: mo.conv_transpose(%[[INPUT]], %[[FILTER]])
  // CHECK-SAME: [strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]], output_paddings = %[[OP]]]
  %0 = mo.conv_transpose(%input, %filter) {filter_layout = ""}
    [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, f32>,
    !mo.tensor<Sh2, si64>, !mo.tensor<[D0], si64>, !mo.tensor<[D1], si64>,  !mo.tensor<[D0], si64>
  ) -> !mo.tensor<Sh3, f32>
  mo.output %0 : !mo.tensor<Sh3, f32>
}

//===----------------------------------------------------------------------===//
// AvgPoolOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @avg_pool(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
mo.graph @avg_pool(%input: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
  // CHECK-DAG: %[[FS:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK: mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]]
  %0 = mo.avg_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[20, 9, 5, 32], f32>
  mo.output %0 : !mo.tensor<[20, 9, 5, 32], f32>
}

// CHECK: mo.graph @avg_pool_dyn_rank<[[SH0:[^,>]*]]: !mosh.ape, [[SH1:[^,>]*]]: !mosh.ape, [[SH2:[^,>]*]]: !mosh.ape, [[SH3:[^,>]*]]: !mosh.ape, [[SH4:[^,>]*]]: !mosh.ape, [[SH5:[^,>]*]]: !mosh.ape>(%[[INPUT:.*]]: !mo.tensor<[[SH0]], f32>, %[[FS:.*]]: !mo.tensor<[[SH1]], si64>, %[[ST:.*]]: !mo.tensor<[[SH2]], si64>, %[[DI:.*]]: !mo.tensor<[[SH3]], si64>, %[[PA:.*]]: !mo.tensor<[[SH4]], si64>) -> !mo.tensor<[[SH5]], f32> {
mo.graph @avg_pool_dyn_rank<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, Sh4: !mosh.ape, Sh5: !mosh.ape>(
    %input: !mo.tensor<Sh0, f32>,
    %fs: !mo.tensor<Sh1, si64>,
    %st: !mo.tensor<Sh2, si64>,
    %di: !mo.tensor<Sh3, si64>,
    %pa: !mo.tensor<Sh4, si64>
  ) -> !mo.tensor<Sh5, f32> {
  // CHECK: mo.avg_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]]
  %0 = mo.avg_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] {count_boundary = false} : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<Sh4, si64>) -> !mo.tensor<Sh5, f32>
  mo.output %0 : !mo.tensor<Sh5, f32>
}

//===----------------------------------------------------------------------===//
// MaxPoolOp
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mo.graph @max_pool(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
mo.graph @max_pool(%input: !mo.tensor<[20, 10, 10, 32], f32>) -> !mo.tensor<[20, 9, 5, 32], f32> {
  // CHECK-DAG: %[[FS:.*]] = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[ST:.*]] = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[DI:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PA:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  %fs = mo.constant {value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %st = mo.constant {value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %di = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  %pa = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK: mo.max_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]]
  %0 = mo.max_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
  ) -> !mo.tensor<[20, 9, 5, 32], f32>
  mo.output %0 : !mo.tensor<[20, 9, 5, 32], f32>
}

// CHECK-LABEL: mo.graph @max_pool_dyn_rank
// CHECK-SAME: <[[SH0:[^,>]*]]: !mosh.ape, [[SH1:[^,>]*]]: !mosh.ape, [[SH2:[^,>]*]]: !mosh.ape, [[SH3:[^,>]*]]: !mosh.ape, [[D0:[^,>]*]], [[SH4:[^,>]*]]: !mosh.ape>(
// CHECK-SAME: %[[INPUT:.*]]: !mo.tensor<[[SH0]], f32>, %[[FS:.*]]: !mo.tensor<[[SH1]], si64>, %[[ST:.*]]: !mo.tensor<[[SH2]], si64>, %[[DI:.*]]: !mo.tensor<[[SH3]], si64>, %[[PA:.*]]: !mo.tensor<{{\[}}[[D0]]{{\]}}, si64>) -> !mo.tensor<[[SH4]], f32> {
mo.graph @max_pool_dyn_rank<Sh0: !mosh.ape, Sh1: !mosh.ape, Sh2: !mosh.ape, Sh3: !mosh.ape, D0, Sh4: !mosh.ape>(
    %input: !mo.tensor<Sh0, f32>,
    %fs: !mo.tensor<Sh1, si64>,
    %st: !mo.tensor<Sh2, si64>,
    %di: !mo.tensor<Sh3, si64>,
    %pa: !mo.tensor<[D0], si64>
  ) -> !mo.tensor<Sh4, f32> {
  // CHECK: mo.max_pool(%[[INPUT]]) [filter_shape = %[[FS]], strides = %[[ST]], dilations = %[[DI]], paddings = %[[PA]]]
  %0 = mo.max_pool(%input) [filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa] : (
    !mo.tensor<Sh0, f32>, !mo.tensor<Sh1, si64>, !mo.tensor<Sh2, si64>, !mo.tensor<Sh3, si64>, !mo.tensor<[D0], si64>) -> !mo.tensor<Sh4, f32>
  mo.output %0 : !mo.tensor<Sh4, f32>
}
