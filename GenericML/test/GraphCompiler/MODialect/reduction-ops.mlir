// RUN: mo-opt %s --canonicalize | FileCheck %s

// CHECK: mo.graph @mean(%[[INPUT:.*]]: !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32> no_inline {
mo.graph @mean(%input: !mo.tensor<[2, 3], f32>) -> (!mo.tensor<[2, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.mean(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
  %output = mo.mean(%input, %axis) : (!mo.tensor<[2, 3], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1], f32>
  mo.output %output : !mo.tensor<[2, 1], f32>
}

// CHECK: mo.graph @mean_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[2, 3, 5], f32>) -> !mo.tensor<[2, 1, 5], f32> no_inline {
mo.graph @mean_negative_index_canonicalization(%input: !mo.tensor<[2, 3, 5], f32>) -> (!mo.tensor<[2, 1, 5], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<1> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.mean(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[2, 3, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1, 5], f32>
  %output = mo.mean(%input, %axis) : (!mo.tensor<[2, 3, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[2, 1, 5], f32>
  mo.output %output : !mo.tensor<[2, 1, 5], f32>
}

// CHECK: mo.graph @reduce_mul(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_mul(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.mul(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.mul(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_mul_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_mul_negative_index_canonicalization(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.mul(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.mul(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_mul_fold() -> !mo.tensor<[1], si64> no_inline {
mo.graph @reduce_mul_fold() -> !mo.tensor<[1], si64> {
  %axis = mo.constant {value = #M.dense_array<0> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %input = mo.constant {value = #M.dense_array<2, 3, 4> : tensor<3xsi64>} : !mo.tensor<[3], si64>
  // CHECK-NEXT: %[[RES:.*]] = mo.constant {value = #M.dense_array<24> : tensor<1xsi64>} : !mo.tensor<[1], si64>
  %output = mo.reduce.mul(%input, %axis) : (!mo.tensor<[3], si64>, !mo.tensor<[], si64>) -> !mo.tensor<[1], si64>
  // CHECK-NEXT: mo.output %[[RES]] : !mo.tensor<[1], si64>
  mo.output %output : !mo.tensor<[1], si64>
}

// CHECK: mo.graph @reduce_add(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_add(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.add(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_add_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[5, ?, ?], f32>) -> !mo.tensor<[1, ?, ?], f32> no_inline {
mo.graph @reduce_add_negative_index_canonicalization(%input: !mo.tensor<[5, ?, ?], f32>) -> (!mo.tensor<[1, ?, ?], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<0> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-3> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.add(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[5, ?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1, ?, ?], f32>
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[5, ?, ?], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[1, ?, ?], f32>
  mo.output %output : !mo.tensor<[1, ?, ?], f32>
}

// CHECK: mo.graph @reduce_add_fold(%[[INPUT:.*]]: !mo.tensor<[?, ?, 1], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_add_fold(%input: !mo.tensor<[?, ?, 1], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: mo.output %[[INPUT:.*]] : !mo.tensor<[?, ?, 1], f32>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %output = mo.reduce.add(%input, %axis) : (!mo.tensor<[?, ?, 1], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_min(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_min(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.min(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.min(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_min_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_min_negative_index_canonicalization(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.min(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.min(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_max(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_max(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.max(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.max(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @reduce_max_negative_index_canonicalization(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], f32> no_inline {
mo.graph @reduce_max_negative_index_canonicalization(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], f32>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : !M.array<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.reduce.max(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  %output = mo.reduce.max(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], f32>
  mo.output %output : !mo.tensor<[?, ?, 1], f32>
}

// CHECK: mo.graph @arg_max(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64> no_inline {
mo.graph @arg_max(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], si64>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.arg_max(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], si64>
  %output = mo.arg_max(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], si64>
  mo.output %output : !mo.tensor<[?, ?, 1], si64>
}

// CHECK: mo.graph @arg_min(%[[INPUT:.*]]: !mo.tensor<[?, ?, 5], f32>) -> !mo.tensor<[?, ?, 1], si64> no_inline {
mo.graph @arg_min(%input: !mo.tensor<[?, ?, 5], f32>) -> (!mo.tensor<[?, ?, 1], si64>) {
  // CHECK: %[[AXIS:.*]] = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  %axis = mo.constant {value = #M.dense_array<2> : tensor<1xsi64>} : !mo.tensor<[], si64>
  // CHECK-NEXT: mo.arg_min(%[[INPUT]], %[[AXIS]]) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], si64>
  %output = mo.arg_min(%input, %axis) : (!mo.tensor<[?, ?, 5], f32>, !mo.tensor<[], si64>) -> !mo.tensor<[?, ?, 1], si64>
  mo.output %output : !mo.tensor<[?, ?, 1], si64>
}
