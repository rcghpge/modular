// RUN: mo-opt --mo-legalize-rmo-operators --canonicalize --split-input-file %s | FileCheck %s --enable-var-scope

// CHECK-LABEL: mo.graph @conv_transpose
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, 2, 2, 1], si64>,
// CHECK:  %[[FILTER:.*]]: !mo.tensor<[2, 2, 1, 1], si64>)
mo.graph @conv_transpose(
  %input: !mo.tensor<[1, 2, 2, 1], si64>,
  %filter: !mo.tensor<[2, 2, 1, 1], si64>
) -> (!mo.tensor<[1, 3, 3, 1], si64>) {

  // CHECK-DAG: %[[C1:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PADDINGS:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[OUT_PADDINGS:.*]] = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[CONV:.*]] = mo.conv_transpose<() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT]], %[[FILTER]]) {filter_layout = ""} [strides = %[[C1]], dilations = %[[C1]], paddings = %[[PADDINGS]], output_paddings = %[[OUT_PADDINGS]]] : (!mo.tensor<[1, 2, 2, 1], si64>, !mo.tensor<[2, 2, 1, 1], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], si64>
  // CHECK: kgen.param.declare [[EQ:.*]]: i1 = <and(eq(:si64 [[D0]], 1), eq(:si64 [[D1]], 3), eq(:si64 [[D2]], 3), eq(:si64 [[D3]], 1))>
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, [[EQ]])>
  // CHECK: %[[OUT:.*]] = mo.rebind(%[[CONV]]) : !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], si64> -> !mo.tensor<[1, 3, 3, 1], si64>

  %output = rmo.conv_transpose(%input, %filter) {
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    strides = #mosh<ape[1, 1]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    output_paddings = #mosh<ape[0, 0]> : !mosh.ape
  } : (
    !mo.tensor<[1, 2, 2, 1], si64>,
    !mo.tensor<[2, 2, 1, 1], si64>
  ) -> !mo.tensor<[1, 3, 3, 1], si64>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[1, 3, 3, 1], si64>
  mo.output %output : !mo.tensor<[1, 3, 3, 1], si64>
}

// -----

// CHECK-LABEL: mo.graph @conv_transpose_symbolic<height, width, channels>
// CHECK: (%[[INPUT:.*]]: !mo.tensor<[1, height, width, channels], si64>,
// CHECK:  %[[FILTER:.*]]: !mo.tensor<[2, 2, 1, 1], si64>)
mo.graph @conv_transpose_symbolic<height, width, channels>(
  %input: !mo.tensor<[1, height, width, channels], si64>,
  %filter: !mo.tensor<[2, 2, 1, 1], si64>
) -> (!mo.tensor<[1, add(height, 1), add(width, 1), 1], si64>) {

  // CHECK-DAG: %[[C1:.*]] = mo.constant {value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK-DAG: %[[PADDINGS:.*]] = mo.constant {value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
  // CHECK-DAG: %[[OUT_PADDINGS:.*]] = mo.constant {value = #M.dense_array<0, 0> : tensor<2xsi64>} : !mo.tensor<[2], si64>
  // CHECK: %[[CONV:.*]] = mo.conv_transpose<() -> [[D0:.*]], [[D1:.*]], [[D2:.*]], [[D3:.*]]>(%[[INPUT]], %[[FILTER]]) {filter_layout = ""} [strides = %[[C1]], dilations = %[[C1]], paddings = %[[PADDINGS]], output_paddings = %[[OUT_PADDINGS]]] : (!mo.tensor<[1, height, width, channels], si64>, !mo.tensor<[2, 2, 1, 1], si64>, !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[2], si64>) -> !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], si64>
  // CHECK: kgen.param.declare [[EQ:.*]]: i1 = <and(eq(:si64 add(height, 1), [[D1]]), eq(:si64 add(width, 1), [[D2]]), eq(:si64 [[D0]], 1), eq(:si64 [[D3]], 1))>
  // CHECK: kgen.param.declare {{.*}}: !mo.chain = <apply(:(i1) -> !mo.chain "mo.assert"{message = "rebind shape check failed" : !kgen.string}, [[EQ]])>
  // CHECK: %[[OUT:.*]] = mo.rebind(%[[CONV]]) : !mo.tensor<[[[D0]], [[D1]], [[D2]], [[D3]]], si64> -> !mo.tensor<[1, add(height, 1), add(width, 1), 1], si64>

  %output = rmo.conv_transpose(%input, %filter) {
    dilations = #mosh<ape[1, 1]> : !mosh.ape,
    strides = #mosh<ape[1, 1]> : !mosh.ape,
    paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
    output_paddings = #mosh<ape[0, 0]> : !mosh.ape
  } : (
    !mo.tensor<[1, height, width, channels], si64>,
    !mo.tensor<[2, 2, 1, 1], si64>
  ) -> !mo.tensor<[1, add(height, 1), add(width, 1), 1], si64>

  // CHECK: mo.output %[[OUT]] : !mo.tensor<[1, add(height, 1), add(width, 1), 1], si64>
  mo.output %output : !mo.tensor<[1, add(height, 1), add(width, 1), 1], si64>
}
