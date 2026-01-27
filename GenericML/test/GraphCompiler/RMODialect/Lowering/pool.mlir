// RUN: mo-opt --mo-legalize-rmo-operators --canonicalize %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @max_pool
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, 6, 15, 1], f32>)
// CHECK: -> !mo.tensor<[1, 2, 7, 1], f32>

mo.graph @max_pool(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 7, 1], f32> {

  // CHECK-DAG: %[[C0:.*]] = mo.constant {value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[C2:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  // CHECK: %[[MAXPOOL:.*]] = mo.max_pool<() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%arg0) [filter_shape = %[[C0]], strides = %[[C0]], dilations = %[[C2]], paddings = %[[C1]]] : (!mo.tensor<[1, 6, 15, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>
  // CHECK: kgen.param.declare [[EQ:.*]]: i1 = <and(eq(:si64 [[D0]], 1), eq(:si64 [[D1]], 2), eq(:si64 [[D2]], 7), eq(:si64 [[D3]], 1))>
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, [[EQ]])>
  // CHECK: %[[OUT:.*]] = mo.rebind(%[[MAXPOOL]]) : !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[1, 2, 7, 1], f32>

  %out = rmo.max_pool(%input) {
    filter_shape = #mosh<ape[3, 2]> : !mosh.ape,
    strides = #mosh<ape[3, 2]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    ceil_mode = false
  }: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 7, 1], f32>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[1, 2, 7, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 7, 1], f32>
}

// -----

// CHECK-LABEL: mo.graph @max_pool_ceil
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, 6, 15, 1], f32>)
// CHECK: -> !mo.tensor<[1, 2, 8, 1], f32>

mo.graph @max_pool_ceil(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 8, 1], f32> {

  // CHECK-DAG: %[[C0:.*]] = mo.constant {value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[C2:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  // CHECK: %[[MAXPOOL:.*]] = mo.max_pool_ceil_mode_true<() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%arg0) [filter_shape = %[[C0]], strides = %[[C0]], dilations = %[[C2]], paddings = %[[C1]]] : (!mo.tensor<[1, 6, 15, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>
  // CHECK: kgen.param.declare [[EQ:.*]]: i1 = <and(eq(:si64 [[D0]], 1), eq(:si64 [[D1]], 2), eq(:si64 [[D2]], 8), eq(:si64 [[D3]], 1))>
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, [[EQ]])>
  // CHECK: %[[OUT:.*]] = mo.rebind(%[[MAXPOOL]]) : !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[1, 2, 8, 1], f32>

  %out = rmo.max_pool(%input) {
    filter_shape = #mosh<ape[3, 2]> : !mosh.ape,
    strides = #mosh<ape[3, 2]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    ceil_mode = true
  }: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 8, 1], f32>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[1, 2, 8, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 8, 1], f32>
}

// -----

// CHECK-LABEL: mo.graph @avg_pool
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, 6, 15, 1], f32>)
// CHECK: -> !mo.tensor<[1, 2, 7, 1], f32>

mo.graph @avg_pool(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 7, 1], f32> {

  // CHECK-DAG: %[[C0:.*]] = mo.constant {value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[C2:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  // CHECK: %[[MAXPOOL:.*]] = mo.avg_pool<() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%arg0) [filter_shape = %[[C0]], strides = %[[C0]], dilations = %[[C2]], paddings = %[[C1]]] {count_boundary = true} : (!mo.tensor<[1, 6, 15, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>
  // CHECK: kgen.param.declare [[EQ:.*]]: i1 = <and(eq(:si64 [[D0]], 1), eq(:si64 [[D1]], 2), eq(:si64 [[D2]], 7), eq(:si64 [[D3]], 1))>
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, [[EQ]])>
  // CHECK: %[[OUT:.*]] = mo.rebind(%[[MAXPOOL]]) : !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[1, 2, 7, 1], f32>

  %out = rmo.avg_pool(%input) {
    filter_shape = #mosh<ape[3, 2]> : !mosh.ape,
    strides = #mosh<ape[3, 2]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    ceil_mode = false,
    count_boundary = true
  }: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 7, 1], f32>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[1, 2, 7, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 7, 1], f32>
}

// -----

// CHECK-LABEL: mo.graph @avg_pool_ceil
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, 6, 15, 1], f32>)
// CHECK: -> !mo.tensor<[1, 2, 8, 1], f32>

mo.graph @avg_pool_ceil(
  %input: !mo.tensor<[1, 6, 15, 1], f32>
) -> !mo.tensor<[1, 2, 8, 1], f32> {

  // CHECK-DAG: %[[C0:.*]] = mo.constant {value = #M.dense_array<3, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[C1:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[C2:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>

  // CHECK: %[[MAXPOOL:.*]] = mo.avg_pool_ceil_mode_true<() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%arg0) [filter_shape = %[[C0]], strides = %[[C0]], dilations = %[[C2]], paddings = %[[C1]]] {count_boundary = true} : (!mo.tensor<[1, 6, 15, 1], f32>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32>
  // CHECK: kgen.param.declare [[EQ:.*]]: i1 = <and(eq(:si64 [[D0]], 1), eq(:si64 [[D1]], 2), eq(:si64 [[D2]], 8), eq(:si64 [[D3]], 1))>
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, [[EQ]])>
  // CHECK: %[[OUT:.*]] = mo.rebind(%[[MAXPOOL]]) : !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], f32> -> !mo.tensor<[1, 2, 8, 1], f32>

  %out = rmo.avg_pool(%input) {
    filter_shape = #mosh<ape[3, 2]> : !mosh.ape,
    strides = #mosh<ape[3, 2]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    ceil_mode = true,
    count_boundary = true
  }: (!mo.tensor<[1, 6, 15, 1], f32>) -> !mo.tensor<[1, 2, 8, 1], f32>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[1, 2, 8, 1], f32>
  mo.output %out : !mo.tensor<[1, 2, 8, 1], f32>
}
