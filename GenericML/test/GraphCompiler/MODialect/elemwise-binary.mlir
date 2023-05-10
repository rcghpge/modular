// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @add(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @add(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.add(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.add(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @div(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @div(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.div(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.div(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @max(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @max(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.max(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.max(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @min(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @min(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.min(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.min(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @mul(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @mul(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.mul(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.mul(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @pow(%[[ARG0:.*]]: ![[ARG0TYPE:.*]], %[[ARG1:.*]]: ![[ARG1TYPE:.*]]) -> !mo.tensor<[2, 3], f32> {
mo.graph @pow(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.pow(%[[ARG0]] : ![[ARG0TYPE]], %[[ARG1]] :
  %0 = mo.pow(%arg0 : !mo.tensor<[2, 3], f32>, %arg1 : !mo.tensor<[2, 3], f32>) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @pow_int_exponent(%[[ARG0:.*]]: ![[ARG0TYPE:.*]], %[[ARG1:.*]]: ![[ARG1TYPE:.*]]) -> !mo.tensor<[2, 3], f32> {
mo.graph @pow_int_exponent(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], si32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.pow(%[[ARG0]] : ![[ARG0TYPE]], %[[ARG1]] : ![[ARG1TYPE]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.pow(%arg0 : !mo.tensor<[2, 3], f32>, %arg1 : !mo.tensor<[2, 3], si32>) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @sub(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32> {
mo.graph @sub(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], f32>) {
  // CHECK-NEXT: mo.sub(%[[ARG0]], %[[ARG1]]) : !mo.tensor<[2, 3], f32>
  %0 = mo.sub(%arg0, %arg1) : !mo.tensor<[2, 3], f32>
  mo.output %0 : !mo.tensor<[2, 3], f32>
}

// CHECK: mo.graph @equal(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @equal(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.equal(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.equal(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @equal_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @equal_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant {value = #M.dense_array<false, true, false, false, false, true> : tensor<2x3xi1>} : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant {value = #M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant {value = #M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %0 = mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @greater(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.greater(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.greater(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @greater_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant {value = #M.dense_array<false, false, true, false, true, false> : tensor<2x3xi1>} : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant {value = #M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant {value = #M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %0 = mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater_equal(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @greater_equal(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.greater_equal(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.greater_equal(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @greater_equal_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @greater_equal_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant {value = #M.dense_array<false, true, true, false, true, true> : tensor<2x3xi1>} : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant {value = #M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant {value = #M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %0 = mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @not_equal(%[[ARG0:.*]]: !mo.tensor<[2, 3], f32>, %[[ARG1:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool> {
mo.graph @not_equal(%arg0: !mo.tensor<[2, 3], f32>, %arg1: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 3], bool>) {
  // CHECK-NEXT: mo.not_equal(%[[ARG0]], %[[ARG1]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  %0 = mo.not_equal(%arg0, %arg1) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}

// CHECK: mo.graph @not_equal_int_folder() -> !mo.tensor<[2, 3], bool> {
mo.graph @not_equal_int_folder() -> (!mo.tensor<[2, 3], bool>) {
  // CHECK: %[[RES:.*]] = mo.constant {value = #M.dense_array<true, false, true, true, true, false> : tensor<2x3xi1>} : !mo.tensor<[2, 3], bool>
  %lhs = mo.constant {value = #M.dense_array<0, -1, 2, -2, 4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %rhs = mo.constant {value = #M.dense_array<2, -1, 1, 4, -4, 5> : tensor<2x3xsi32>} : !mo.tensor<[2, 3], si32>
  %0 = mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], si32>, !mo.tensor<[2, 3], si32>) -> !mo.tensor<[2, 3], bool>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[2, 3], bool>
  mo.output %0 : !mo.tensor<[2, 3], bool>
}
